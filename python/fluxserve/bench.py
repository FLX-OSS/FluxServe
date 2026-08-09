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


from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import sys
import time
import traceback
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import aiohttp
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from fluxserve.prompt_utils import render_openai_messages


DEFAULT_TIMEOUT_SEC = 60 * 60
SUPPORTED_METRICS = ("E2E", "QUEUE", "EXECUTION", "HTTP_OVERHEAD")
METRIC_RESULT_NAMES = {
    "E2E": "e2e",
    "QUEUE": "queue",
    "EXECUTION": "execution",
    "HTTP_OVERHEAD": "http_overhead",
}


@dataclass
class SampleRequest:
    messages: list[dict[str, str]]
    prompt_len: int
    output_len: int
    request_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestOutput:
    success: bool = False
    generated_text: str = ""
    client_latency: float = 0.0
    server_e2e_latency: float | None = None
    queue_latency: float | None = None
    execution_latency: float | None = None
    engine_e2e_latency: float | None = None
    prompt_len: int = 0
    output_tokens: int = 0
    server_prompt_tokens: int | None = None
    server_completion_tokens: int | None = None
    start_time: float = 0.0
    error: str = ""
    retry_count: int = 0

    @property
    def e2e_latency(self) -> float:
        if self.server_e2e_latency is not None:
            return self.server_e2e_latency
        return self.client_latency

    @property
    def e2e_source(self) -> str:
        return "server" if self.server_e2e_latency is not None else "client_fallback"


def _print_metric(label: str, value: Any, precision: int | None = None) -> None:
    if precision is None:
        print(f"{label:<42} {value}")
    else:
        print(f"{label:<42} {value:.{precision}f}")


def _set_ulimit(target_soft_limit: int = 65535) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < target_soft_limit:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(target_soft_limit, hard), hard))
        except ValueError:
            pass


def _join_host_port(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))


def _parse_metrics(value: str) -> tuple[str, ...]:
    requested = {item.strip().upper() for item in value.split(",") if item.strip()}
    if not requested:
        raise argparse.ArgumentTypeError("--metrics must contain at least E2E.")
    unknown = requested.difference(SUPPORTED_METRICS)
    if unknown:
        supported = ", ".join(SUPPORTED_METRICS)
        raise argparse.ArgumentTypeError(
            f"Unknown metric(s): {', '.join(sorted(unknown))}. Supported metrics: {supported}."
        )
    if "E2E" not in requested:
        raise argparse.ArgumentTypeError("--metrics must include E2E.")
    return tuple(metric for metric in SUPPORTED_METRICS if metric in requested)


def _create_benchmark_connector(max_concurrency: int | None = None) -> aiohttp.TCPConnector:
    return aiohttp.TCPConnector(
        limit=max_concurrency or 0,
        force_close=False,
        keepalive_timeout=30,
    )


class SSEDecoder:
    def __init__(self) -> None:
        self.buffer = ""

    def add(self, chunk: bytes) -> list[str]:
        self.buffer += chunk.decode("utf-8", errors="replace")
        messages = []
        while "\n\n" in self.buffer:
            message, self.buffer = self.buffer.split("\n\n", 1)
            message = message.strip()
            if message:
                messages.append(message)
        return messages


def load_jsonl_requests(
    dataset: str,
    tokenizer: PreTrainedTokenizerBase,
    num_prompts: int | None,
    default_output_len: int,
    output_len_override: int | None,
    request_id_prefix: str,
) -> list[SampleRequest]:
    if num_prompts is not None and num_prompts <= 0:
        raise ValueError("--num-prompts must be positive.")
    if output_len_override is not None and output_len_override <= 0:
        raise ValueError("--dataset-output-len must be positive.")

    rows: list[tuple[int, dict[str, Any]]] = []
    with open(dataset, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {dataset}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {dataset}:{line_number} must be an object.")
            rows.append((line_number, row))

    if not rows:
        raise ValueError(f"Dataset is empty: {dataset}")
    if num_prompts is not None and num_prompts > len(rows):
        raise ValueError(
            f"--num-prompts={num_prompts} exceeds dataset size {len(rows)}."
        )

    requests = []
    seen_ids: set[str] = set()
    selected_rows = rows if num_prompts is None else rows[:num_prompts]
    for dataset_index, (line_number, row) in enumerate(selected_rows):
        messages = _validate_messages(row.get("messages"), dataset, line_number)
        metadata = row.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata in {dataset}:{line_number} must be an object.")
        logical_id = str(metadata.get("task_id", dataset_index))
        request_id = f"{request_id_prefix}{logical_id}"
        if request_id in seen_ids:
            raise ValueError(f"Duplicate request ID {request_id!r} in {dataset}:{line_number}.")
        seen_ids.add(request_id)

        body = {key: value for key, value in row.items() if key not in ("messages", "metadata")}
        output_len = output_len_override
        if output_len is None:
            output_len = body.get("max_tokens", default_output_len)
        if isinstance(output_len, bool) or not isinstance(output_len, int) or output_len <= 0:
            raise ValueError(f"max_tokens in {dataset}:{line_number} must be a positive integer.")
        body["max_tokens"] = output_len
        # Measure the exact prompt sent through the online server.  This must
        # stay aligned with bench_offline and the chat endpoint.
        input_ids = tokenizer(render_openai_messages(messages))["input_ids"]
        requests.append(
            SampleRequest(
                messages=messages,
                prompt_len=len(input_ids),
                output_len=output_len,
                request_id=request_id,
                metadata=dict(metadata),
                body=body,
            )
        )

    print(f"#Input tokens: {sum(req.prompt_len for req in requests)}")
    print(f"#Output tokens: {sum(req.output_len for req in requests)}")
    return requests


def _validate_messages(messages, dataset: str, line_number: int) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"messages in {dataset}:{line_number} must be a nonempty list.")
    validated = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"messages[{index}] in {dataset}:{line_number} must be an object."
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(
                f"messages[{index}].role in {dataset}:{line_number} must be nonempty."
            )
        if not isinstance(content, str):
            raise ValueError(
                f"messages[{index}].content in {dataset}:{line_number} must be a string."
            )
        validated.append({"role": role, "content": content})
    return validated


async def iter_requests(
    requests: list[SampleRequest],
    request_rate: float,
    burstiness: float,
) -> AsyncIterator[SampleRequest]:
    if request_rate <= 0:
        raise ValueError("--request-rate must be positive.")

    for i, request in enumerate(requests):
        if i and request_rate != float("inf"):
            await asyncio.sleep(float(np.random.exponential(1.0 / request_rate)))
        yield request


async def wait_for_health(base_url: str, timeout_s: int) -> None:
    if timeout_s <= 0:
        return
    deadline = time.perf_counter() + timeout_s
    health_url = f"{base_url}/health"
    last_error = ""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        while time.perf_counter() < deadline:
            try:
                async with session.get(health_url) as response:
                    if response.status == 200:
                        return
                    last_error = f"HTTP {response.status}"
            except Exception as exc:
                last_error = str(exc)
            await asyncio.sleep(1)
    raise RuntimeError(f"Server did not become healthy at {health_url}: {last_error}")


async def send_request(
    session: aiohttp.ClientSession,
    api_url: str,
    model: str,
    request: SampleRequest,
    tokenizer: PreTrainedTokenizerBase | None,
    extra_body: dict[str, Any],
    ignore_eos: bool,
) -> RequestOutput:
    payload = dict(request.body)
    payload.update(
        {
        "model": model,
        "messages": request.messages,
        "stream": True,
        }
    )
    if ignore_eos:
        payload["ignore_eos"] = True
    payload.update(extra_body)
    headers = {
        "Content-Type": "application/json",
        "x-request-id": request.request_id,
        "x-idempotency-key": request.request_id,
    }
    started = time.perf_counter()
    retryable = (
        aiohttp.ClientConnectionError,
        aiohttp.ClientPayloadError,
        ConnectionResetError,
    )

    async def attempt(active_session: aiohttp.ClientSession, retry_count: int):
        output = RequestOutput(
            prompt_len=request.prompt_len,
            start_time=started,
            retry_count=retry_count,
        )
        generated_text = ""
        async with active_session.post(api_url, json=payload, headers=headers) as response:
            if response.status != 200:
                response_text = await response.text()
                output.error = f"HTTP {response.status}: {response_text}"
                return output

            decoder = SSEDecoder()
            saw_chunk = False
            async for raw in response.content.iter_any():
                if not raw.strip():
                    continue
                for message in decoder.add(raw):
                    if not message.startswith("data: "):
                        continue
                    data_text = message.removeprefix("data: ").strip()
                    if data_text == "[DONE]":
                        continue
                    data = json.loads(data_text)
                    if "error" in data:
                        output.error = str(data["error"])
                        return output
                    if data.get("object") == "fluxserve.metrics":
                        _parse_server_metrics(output, data)
                        continue
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    text = choice.get("text", "")
                    if not text:
                        message = choice.get("message") or {}
                        delta = choice.get("delta") or {}
                        text = message.get("content", "") or delta.get("content", "")
                    saw_chunk = True
                    generated_text += text or ""

            output.success = saw_chunk
            output.generated_text = generated_text
            output.client_latency = time.perf_counter() - started
            if tokenizer is not None:
                output.output_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
            else:
                output.output_tokens = request.output_len
            if not output.success:
                output.error = "No completion chunk received."
            return output

    try:
        return await attempt(session, 0)
    except retryable:
        connector = aiohttp.TCPConnector(limit=1, force_close=True)
        async with aiohttp.ClientSession(
            timeout=session.timeout,
            connector=connector,
        ) as retry_session:
            try:
                return await attempt(retry_session, 1)
            except Exception:
                output = RequestOutput(
                    prompt_len=request.prompt_len,
                    start_time=started,
                    retry_count=1,
                )
                output.error = "".join(traceback.format_exception(*sys.exc_info()))
                return output
    except Exception:
        output = RequestOutput(prompt_len=request.prompt_len, start_time=started)
        output.error = "".join(traceback.format_exception(*sys.exc_info()))
        return output


def _parse_server_metrics(output: RequestOutput, data: dict[str, Any]) -> None:
    float_fields = {
        "server_e2e_latency_s": "server_e2e_latency",
        "queue_latency_s": "queue_latency",
        "execution_latency_s": "execution_latency",
        "e2e_latency_s": "engine_e2e_latency",
    }
    for wire_name, attr_name in float_fields.items():
        value = data.get(wire_name)
        if isinstance(value, (int, float)) and value >= 0:
            setattr(output, attr_name, float(value))
    int_fields = {
        "prompt_tokens": "server_prompt_tokens",
        "completion_tokens": "server_completion_tokens",
    }
    for wire_name, attr_name in int_fields.items():
        value = data.get(wire_name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            setattr(output, attr_name, value)


def summarize(
    outputs: list[RequestOutput],
    duration_s: float,
    percentiles: list[float],
    metrics: tuple[str, ...] = ("E2E",),
) -> dict[str, Any]:
    successes = [output for output in outputs if output.success]
    failed = [output for output in outputs if not output.success]
    input_tokens = [
        output.server_prompt_tokens
        if output.server_prompt_tokens is not None
        else output.prompt_len
        for output in successes
    ]
    completion_tokens = [
        output.server_completion_tokens
        if output.server_completion_tokens is not None
        else output.output_tokens
        for output in successes
    ]
    total_input = sum(input_tokens)
    # Throughput must reflect tokens actually emitted by the client-visible
    # response.  Server-reported completion_tokens can be model-internal
    # diffusion/block tokens and may substantially exceed generated text.
    actual_output = sum(output.output_tokens for output in successes)
    arrival_times = [output.start_time for output in successes]
    arrival_window_s = (
        max(arrival_times) - min(arrival_times) if len(arrival_times) >= 2 else 0.0
    )
    all_latency_values = {
        "e2e": [output.e2e_latency * 1000 for output in successes],
        "queue": [output.queue_latency * 1000 for output in successes if output.queue_latency is not None],
        "execution": [output.execution_latency * 1000 for output in successes if output.execution_latency is not None],
        "http_overhead": [
            max(0.0, output.client_latency - output.e2e_latency) * 1000
            for output in successes
        ],
    }

    result = {
        "duration": duration_s,
        "completed": len(successes),
        "failed": len(failed),
        "total_input_tokens": total_input,
        "max_output_tokens": actual_output,
        "request_throughput": len(successes) / duration_s if duration_s else 0.0,
        "input_token_throughput": (
            total_input / arrival_window_s if arrival_window_s > 0 else 0.0
        ),
        "input_arrival_window_s": arrival_window_s,
        "output_token_throughput": actual_output / duration_s if duration_s else 0.0,
        "total_token_throughput": (total_input + actual_output) / duration_s if duration_s else 0.0,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": [output.output_tokens for output in outputs],
        "start_times": [output.start_time for output in outputs],
        "e2els": [output.e2e_latency for output in outputs],
        "engine_e2els": [output.engine_e2e_latency for output in outputs],
        "client_latencies": [output.client_latency for output in outputs],
        "retry_counts": [output.retry_count for output in outputs],
        "total_retries": sum(output.retry_count for output in outputs),
        "metrics": list(metrics),
        "e2e_sources": [output.e2e_source for output in outputs],
        "token_count_sources": [
            "server"
            if output.server_completion_tokens is not None
            else "client_fallback"
            for output in outputs
        ],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    if "QUEUE" in metrics:
        result["queue_latencies"] = [output.queue_latency for output in outputs]
    if "EXECUTION" in metrics:
        result["execution_latencies"] = [output.execution_latency for output in outputs]
    if "HTTP_OVERHEAD" in metrics:
        result["http_overheads"] = [
            max(0.0, output.client_latency - output.e2e_latency)
            for output in outputs
        ]

    latency_values = {
        METRIC_RESULT_NAMES[metric]: all_latency_values[METRIC_RESULT_NAMES[metric]]
        for metric in metrics
    }
    for metric_name, values in latency_values.items():
        result[f"mean_{metric_name}_ms"] = float(np.mean(values)) if values else 0.0
        result[f"median_{metric_name}_ms"] = float(np.median(values)) if values else 0.0
        result[f"std_{metric_name}_ms"] = float(np.std(values)) if values else 0.0
        for percentile in percentiles:
            p_label = str(int(percentile)) if percentile.is_integer() else str(percentile)
            result[f"p{p_label}_{metric_name}_ms"] = _percentile(values, percentile)

    return result


async def run_serving_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    _set_ulimit()
    np.random.seed(args.seed)

    if args.endpoint != "/v1/chat/completions":
        raise ValueError(
            "Dataset benchmarks require --endpoint /v1/chat/completions."
        )
    base_url = args.base_url or f"http://{_join_host_port(args.host, args.port)}"
    api_url = f"{base_url}{args.endpoint}"
    tokenizer_id = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=args.trust_remote_code,
    )

    requests = load_jsonl_requests(
        dataset=args.dataset,
        tokenizer=tokenizer,
        num_prompts=args.num_prompts,
        default_output_len=128,
        output_len_override=args.dataset_output_len,
        request_id_prefix=f"{args.request_id_prefix}{uuid.uuid4().hex[:8]}-",
    )

    await wait_for_health(base_url, args.ready_check_timeout_sec)

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = _create_benchmark_connector(args.max_concurrency)
    semaphore = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def limited_send(request: SampleRequest) -> RequestOutput:
            if semaphore is None:
                return await send_request(
                    session,
                    api_url,
                    args.model,
                    request,
                    tokenizer,
                    args.extra_body,
                    args.ignore_eos,
                )
            async with semaphore:
                return await send_request(
                    session,
                    api_url,
                    args.model,
                    request,
                    tokenizer,
                    args.extra_body,
                    args.ignore_eos,
                )

        print("Starting main benchmark run...")
        start = time.perf_counter()
        tasks = []
        async for request in iter_requests(requests, args.request_rate, args.burstiness):
            tasks.append(asyncio.create_task(limited_send(request)))
        outputs = await asyncio.gather(*tasks)
        duration_s = time.perf_counter() - start

    percentiles = [float(item) for item in args.metric_percentiles.split(",")]
    result = summarize(outputs, duration_s, percentiles, args.metrics)
    result.update(
        {
            "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "model_id": args.model,
            "tokenizer_id": tokenizer_id,
            "dataset": args.dataset,
            "request_mode": "chat",
            "num_prompts": len(requests),
            "request_rate": args.request_rate if args.request_rate < float("inf") else "inf",
            "burstiness": args.burstiness,
            "max_concurrency": args.max_concurrency,
            "request_ids": [request.request_id for request in requests],
            "request_metadata": [request.metadata for request in requests],
            "requested_output_lens": [request.output_len for request in requests],
        }
    )

    print("=" * 50)
    _print_metric("Successful requests:", result["completed"])
    _print_metric("Failed requests:", result["failed"])
    _print_metric("Benchmark duration (s):", result["duration"], 2)
    _print_metric("Request throughput (req/s):", result["request_throughput"], 2)
    _print_metric("Input token throughput (tok/s):", result["input_token_throughput"], 2)
    _print_metric("Output token throughput (tok/s):", result["output_token_throughput"], 2)
    _print_metric("Total token throughput (tok/s):", result["total_token_throughput"], 2)
    for selected_metric in args.metrics:
        metric = METRIC_RESULT_NAMES[selected_metric]
        print("-" * 50)
        _print_metric(f"Mean {metric.upper()} (ms):", result[f"mean_{metric}_ms"], 2)
        _print_metric(f"Median {metric.upper()} (ms):", result[f"median_{metric}_ms"], 2)
        for percentile in percentiles:
            p_label = str(int(percentile)) if percentile.is_integer() else str(percentile)
            _print_metric(f"P{p_label} {metric.upper()} (ms):", result[f"p{p_label}_{metric}_ms"], 2)
    print("=" * 50)

    if not args.save_detailed:
        for key in (
            "input_lens",
            "output_lens",
            "start_times",
            "queue_latencies",
            "execution_latencies",
            "e2els",
            "engine_e2els",
            "client_latencies",
            "http_overheads",
            "retry_counts",
            "e2e_sources",
            "token_count_sources",
            "generated_texts",
            "errors",
            "request_ids",
            "request_metadata",
            "requested_output_lens",
        ):
            result.pop(key, None)

    if args.save_result or args.output_file:
        output_file = args.output_file
        if output_file is None:
            os.makedirs(args.result_dir, exist_ok=True)
            output_file = os.path.join(
                args.result_dir,
                f"fluxserve-{len(requests)}req-{result['date']}.json",
            )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved benchmark result to {output_file}")

    if result["failed"]:
        print("Failed requests during benchmark run detected (capping to 10):")
        for i, error in enumerate([output.error for output in outputs if not output.success][:10]):
            print(f"Error {i}: {error}")

    return result


def add_bench_subparser(subparsers: argparse._SubParsersAction) -> None:
    bench = subparsers.add_parser("bench")
    bench_sub = bench.add_subparsers(dest="bench_type", required=True)
    serve = bench_sub.add_parser("serve")
    serve.add_argument("--model", required=True)
    serve.add_argument("--dataset", required=True)
    serve.add_argument("--tokenizer", default=None)
    serve.add_argument("--base-url", default=None)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--endpoint", default="/v1/chat/completions")
    serve.add_argument("--num-prompts", type=int, default=None)
    serve.add_argument("--dataset-output-len", type=int, default=None)
    serve.add_argument("--request-rate", type=float, default=float("inf"))
    serve.add_argument("--burstiness", type=float, default=1.0)
    serve.add_argument("--max-concurrency", type=int, default=None)
    serve.add_argument("--ready-check-timeout-sec", type=int, default=600)
    serve.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC)
    serve.add_argument("--seed", type=int, default=0)
    serve.add_argument("--request-id-prefix", default="bench-")
    serve.add_argument("--trust-remote-code", action="store_true", default=True)
    serve.add_argument("--extra-body", type=json.loads, default={})
    serve.add_argument("--ignore-eos", action="store_true")
    serve.add_argument("--metric-percentiles", default="50,90,95,99")
    serve.add_argument(
        "--metrics",
        type=_parse_metrics,
        default=("E2E",),
        help="Comma-separated output metrics; E2E is required (default: E2E).",
    )
    serve.add_argument("--save-result", action="store_true")
    serve.add_argument("--save-detailed", action="store_true")
    serve.add_argument("--result-dir", default="bench_results")
    serve.add_argument("--output-file", default=None)
    serve.set_defaults(dispatch_function=lambda args: asyncio.run(run_serving_benchmark(args)))
