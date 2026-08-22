import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest


MODEL_ID = "google/diffusiongemma-26B-A4B-it"
RUN_SERVER_SMOKE = "FLUXSERVE_RUN_DIFFUSION_GEMMA_SERVER"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_json(url: str, payload=None, timeout: float = 120):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        url,
        data=data,
        headers={"content-type": "application/json"} if data else {},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _wait_ready(base_url: str, process: subprocess.Popen, timeout: float = 600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"server exited during startup with code {process.returncode}")
        try:
            status, body = _request_json(f"{base_url}/health", timeout=2)
            if status == 200 and body == {"status": "ok"}:
                return
        except (URLError, TimeoutError):
            pass
        time.sleep(1)
    raise AssertionError("timed out waiting for Diffusion-Gemma server readiness")


def _disconnect_stream(port: int):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.request(
        "POST",
        "/v1/completions",
        body=json.dumps(
            {
                "model": MODEL_ID,
                "prompt": "Write a detailed explanation of diffusion language models.",
                "max_tokens": 64,
                "ignore_eos": True,
                "stream": True,
            }
        ),
        headers={"content-type": "application/json"},
    )
    response = connection.getresponse()
    assert response.status == 200
    connection.close()


@pytest.mark.skipif(
    os.environ.get(RUN_SERVER_SMOKE) != "1",
    reason=f"set {RUN_SERVER_SMOKE}=1 to run the official checkpoint server",
)
def test_official_diffusion_gemma_single_gpu_server(tmp_path):
    model_name = os.environ.get("DIFFUSION_GEMMA_MODEL", MODEL_ID)
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = Path(tmp_path) / "server.log"
    command = [
        sys.executable,
        "-m",
        "fluxserve.cli",
        "serve",
        "--model",
        model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        "1",
        "--dp-size",
        "1",
        "--ep-size",
        "1",
        "--max-num-seqs",
        "2",
        "--max-model-len",
        "128",
        "--max-new-tokens",
        "8",
        "--block-length",
        "8",
        "--max-denoising-steps",
        "2",
        "--attention-backend",
        "sdpa",
        "--kv-cache-layout",
        "dense",
    ]

    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            _wait_ready(base_url, process)

            payloads = [
                {"model": model_name, "prompt": "Name a prime number.", "max_tokens": 8},
                {
                    "model": model_name,
                    "prompt": "Explain in two short sentences why the sky appears blue during the day.",
                    "max_tokens": 8,
                },
            ]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(
                    pool.map(
                        lambda payload: _request_json(
                            f"{base_url}/v1/completions", payload
                        ),
                        payloads,
                    )
                )
            for status, body in responses:
                assert status == 200, body
                assert body["object"] == "text_completion"
                assert isinstance(body["choices"][0]["text"], str)
                assert body["choices"][0]["finish_reason"] in {"length", "stop"}

            status, body = _request_json(
                f"{base_url}/v1/chat/completions",
                {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Explain why the sky is blue in one sentence.",
                        }
                    ],
                    "max_tokens": 8,
                },
            )
            assert status == 200, body
            assert body["object"] == "chat.completion"
            assert isinstance(body["choices"][0]["message"]["content"], str)

            status, body = _request_json(
                f"{base_url}/v1/completions",
                {"model": model_name, "input_ids": [2] * 128, "max_tokens": 8},
            )
            assert status == 500
            assert "exceeds max_model_len=128" in body["error"]

            status, metrics_before = _request_json(f"{base_url}/metrics")
            assert status == 200
            _disconnect_stream(port)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                _, metrics_after = _request_json(f"{base_url}/metrics")
                if metrics_after["aborted_requests"] > metrics_before["aborted_requests"]:
                    break
                time.sleep(0.25)
            else:
                raise AssertionError("disconnected streaming request was not aborted")

            for prompt in ("Say hello.", "Return the number two."):
                status, body = _request_json(
                    f"{base_url}/v1/completions",
                    {"model": model_name, "prompt": prompt, "max_tokens": 4},
                )
                assert status == 200, body
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)

    assert process.returncode in {0, -signal.SIGTERM}, log_path.read_text()
