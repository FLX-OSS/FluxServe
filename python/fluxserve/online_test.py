# Copyright (c) 2026 FLUX-OSS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
    Replay JSONL requests against a running FluxServe HTTP endpoint (debug only).

    Each input row is a JSON object containing the request body (normally at least
    ``messages``). ``request_id`` (or ``id``) is forwarded when present; otherwise
    an ID based on the line number is generated.  One JSON object is written for
    every input row, including failed requests.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiohttp


async def _request(session: aiohttp.ClientSession, url: str, payload: dict[str, Any], request_id: str) -> str:
    headers = {"Content-Type": "application/json", "x-request-id": request_id,
               "x-idempotency-key": request_id}
    async with session.post(url, json=payload, headers=headers) as response:
        if response.status != 200:
            response_text = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {response_text}")
        # The server commonly streams SSE. Also accept a regular JSON response.
        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" not in content_type:
            data = await response.json(content_type=None)
            choices = data.get("choices", []) if isinstance(data, dict) else []
            if choices:
                choice = choices[0]
                return choice.get("text", "") or (choice.get("message") or {}).get("content", "")
            return ""
        answer: list[str] = []
        buffer = ""
        async for raw in response.content:
            buffer += raw.decode("utf-8", errors="replace")
            lines = buffer.split("\n")
            buffer = lines.pop()
            for line in lines:
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if not value or value == "[DONE]":
                    continue
                data = json.loads(value)
                if "error" in data:
                    raise RuntimeError(str(data["error"]))
                for choice in data.get("choices", []):
                    answer.append(choice.get("text", "") or (choice.get("delta") or {}).get("content", ""))
        if buffer.strip().startswith("data:"):
            value = buffer[5:].strip()
            if value and value != "[DONE]":
                data = json.loads(value)
                for choice in data.get("choices", []):
                    answer.append(choice.get("text", "") or (choice.get("delta") or {}).get("content", ""))
        return "".join(answer)


async def _run_one(session, url, line_number, line, args):
    start = time.time()
    request_id = f"{args.request_id_prefix}{line_number}"
    result = {"request_id": request_id, "start_time": start}
    try:
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("input row must be a JSON object")
        request_id = str(row.get("request_id", row.get("id", request_id)))
        payload = dict(row)
        payload.pop("request_id", None)
        payload.pop("id", None)
        result["request_id"] = request_id
        result["answer"] = await _request(session, args.url, payload, request_id)
    except Exception as exc:
        result["answer"] = None
        result["error"] = str(exc)
    result["end_time"] = time.time()
    return result


async def replay(args: argparse.Namespace) -> None:
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        with Path(args.input).open(encoding="utf-8") as source, Path(args.output).open("w", encoding="utf-8") as sink:
            rows = [(n, line) for n, line in enumerate(source, 1) if line.strip()]
            for offset in range(0, len(rows), args.batch_size):
                batch = rows[offset : offset + args.batch_size]
                results = await asyncio.gather(*(_run_one(session, args.url, n, line, args) for n, line in batch))
                for result in results:
                    sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                sink.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--request-id-prefix", default="online-")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of concurrent requests per batch")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    asyncio.run(replay(args))


if __name__ == "__main__":
    main()
