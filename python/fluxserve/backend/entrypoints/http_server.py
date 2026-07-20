# Copyright (c) 2026 FLUX-OSS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from collections.abc import AsyncIterator

from fluxserve.backend.engine import AsyncLLM, GenerateReqInput


@dataclass
class _ReplayEntry:
    payload_hash: str
    events: list[str] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    finished: bool = False
    completed_at: float = 0.0


class RequestReplayRegistry:
    def __init__(self, *, ttl_s: float = 60, max_entries: int = 1024, max_bytes: int = 64 << 20):
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries: OrderedDict[str, _ReplayEntry] = OrderedDict()

    def acquire(self, key: str, payload_hash: str) -> tuple[_ReplayEntry, bool]:
        self._evict()
        entry = self.entries.get(key)
        if entry is not None:
            if entry.payload_hash != payload_hash:
                raise ValueError("idempotency key was already used with a different payload")
            self.entries.move_to_end(key)
            return entry, False
        entry = _ReplayEntry(payload_hash=payload_hash)
        self.entries[key] = entry
        return entry, True

    def publish(self, entry: _ReplayEntry, event: str) -> None:
        entry.events.append(event)
        for subscriber in list(entry.subscribers):
            subscriber.put_nowait(event)

    def finish(self, entry: _ReplayEntry) -> None:
        entry.finished = True
        entry.completed_at = time.monotonic()
        for subscriber in list(entry.subscribers):
            subscriber.put_nowait(None)
        self._evict()

    def _evict(self) -> None:
        now = time.monotonic()
        for key, entry in list(self.entries.items()):
            if entry.finished and now - entry.completed_at > self.ttl_s:
                self.entries.pop(key, None)
        total_bytes = sum(
            len(event.encode("utf-8"))
            for entry in self.entries.values()
            for event in entry.events
        )
        while self.entries and (
            len(self.entries) > self.max_entries or total_bytes > self.max_bytes
        ):
            key, entry = next(iter(self.entries.items()))
            if not entry.finished:
                break
            self.entries.pop(key)
            total_bytes -= sum(len(event.encode("utf-8")) for event in entry.events)


def _completion_payload(model: str, rid: str, text: str, finish_reason: str | None):
    return {
        "id": rid,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": finish_reason,
            }
        ],
    }


def _chat_payload(model: str, rid: str, text: str, finish_reason: str | None):
    return {
        "id": rid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }


def _server_timing_payload(
    received_at: float,
    engine_meta: dict | None = None,
) -> dict[str, str | float | int | None]:
    payload = {
        "object": "fluxserve.metrics",
        "server_e2e_latency_s": time.perf_counter() - received_at,
    }
    if engine_meta:
        payload.update(engine_meta)
    return payload


def _messages_to_prompt(messages, tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant:"


def create_app(engine: AsyncLLM):
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("HTTP serving requires fastapi and uvicorn.") from exc

    app = FastAPI()
    replay_registry = RequestReplayRegistry()

    @app.on_event("startup")
    async def _startup():
        await engine.start()

    @app.on_event("shutdown")
    async def _shutdown():
        await engine.shutdown()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    async def stream_completion(
        req: GenerateReqInput,
        model: str,
        chat: bool,
        received_at: float,
    ) -> AsyncIterator[str]:
        engine_meta = None
        async for output in engine.generate_request(req):
            if output.meta:
                engine_meta = output.meta
            if output.error:
                yield "data: " + json.dumps({"error": output.error}) + "\n\n"
                break
            if chat:
                payload = _chat_payload(model, output.rid, output.text, output.finish_reason)
            else:
                payload = _completion_payload(model, output.rid, output.text, output.finish_reason)
            yield "data: " + json.dumps(payload) + "\n\n"
        yield "data: " + json.dumps(
            _server_timing_payload(received_at, engine_meta)
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    async def replayable_stream(key, payload_hash, stream_factory):
        try:
            entry, owner = replay_registry.acquire(key, payload_hash)
        except ValueError as exc:
            yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"
            return

        if owner:
            async def produce():
                try:
                    async for event in stream_factory():
                        replay_registry.publish(entry, event)
                except Exception as exc:  # noqa: BLE001
                    replay_registry.publish(
                        entry, "data: " + json.dumps({"error": str(exc)}) + "\n\n"
                    )
                finally:
                    replay_registry.finish(entry)

            asyncio.create_task(produce())

        for event in entry.events:
            yield event
        if entry.finished:
            return
        queue = asyncio.Queue()
        entry.subscribers.append(queue)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            if queue in entry.subscribers:
                entry.subscribers.remove(queue)

    def stream_response(request, body, req, model, chat, received_at):
        key = request.headers.get("x-idempotency-key")
        if not key:
            return StreamingResponse(
                stream_completion(req, model, chat=chat, received_at=received_at),
                media_type="text/event-stream",
            )
        payload_hash = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        existing = replay_registry.entries.get(key)
        if existing is not None and existing.payload_hash != payload_hash:
            return JSONResponse(
                {"error": "idempotency key was already used with a different payload"},
                status_code=409,
            )
        return StreamingResponse(
            replayable_stream(
                key,
                payload_hash,
                lambda: stream_completion(req, model, chat=chat, received_at=received_at),
            ),
            media_type="text/event-stream",
        )

    @app.post("/v1/completions")
    async def completions(request: Request):
        received_at = time.perf_counter()
        body = await request.json()
        prompt = body.get("prompt")
        input_ids = body.get("input_ids")
        stream = bool(body.get("stream", False))
        model = body.get("model", engine.server_args.model_name)
        params = {
            "max_tokens": body.get("max_tokens", body.get("max_new_tokens", 128)),
            "ignore_eos": bool(body.get("ignore_eos", False)),
        }
        rid = request.headers.get("x-request-id")
        req = GenerateReqInput(
            text=prompt, input_ids=input_ids, sampling_params=params, stream=stream, rid=rid
        )
        if stream:
            return stream_response(request, body, req, model, False, received_at)
        final_text = ""
        rid = ""
        finish_reason = None
        async for output in engine.generate_request(req):
            if output.error:
                return JSONResponse({"error": output.error}, status_code=500)
            rid = output.rid
            final_text += output.text
            finish_reason = output.finish_reason
        return _completion_payload(model, rid, final_text, finish_reason)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        received_at = time.perf_counter()
        body = await request.json()
        messages = body.get("messages")
        if not isinstance(messages, list):
            return JSONResponse({"error": "messages must be a list"}, status_code=400)
        prompt = _messages_to_prompt(messages, engine.tokenizer)
        stream = bool(body.get("stream", False))
        model = body.get("model", engine.server_args.model_name)
        params = {
            "max_tokens": body.get("max_tokens", body.get("max_new_tokens", 128)),
            "ignore_eos": bool(body.get("ignore_eos", False)),
        }
        rid = request.headers.get("x-request-id")
        req = GenerateReqInput(text=prompt, sampling_params=params, stream=stream, rid=rid)
        if stream:
            return stream_response(request, body, req, model, True, received_at)
        final_text = ""
        rid = ""
        finish_reason = None
        async for output in engine.generate_request(req):
            if output.error:
                return JSONResponse({"error": output.error}, status_code=500)
            rid = output.rid
            final_text += output.text
            finish_reason = output.finish_reason
        return _chat_payload(model, rid, final_text, finish_reason)

    return app


def run(engine: AsyncLLM, host: str, port: int):
    import uvicorn

    uvicorn.run(create_app(engine), host=host, port=port, timeout_keep_alive=30)
