"""Opt-in real HTTP generation, paged scheduling, EOS, and slot reuse."""
import concurrent.futures
import json
import os
import re
import subprocess
import sys

import pytest

from test_diffusion_gemma_server import _free_port, _request_json, _wait_ready


@pytest.mark.skipif(os.getenv("FLUXSERVE_RUN_FA4_E2E") != "1", reason="requires GH200 and LLaDA2.1-mini weights")
@pytest.mark.parametrize("decoder", ["threshold", "hierarchy"])
def test_fa4_llada_http_generation(tmp_path, decoder):
    model = os.getenv("LLADA2_FA4_MODEL", "inclusionAI/LLaDA2.1-mini")
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    command = [sys.executable, "-m", "fluxserve.cli", "serve", "--model", model,
               "--host", "127.0.0.1", "--port", str(port), "--max-num-seqs", "2",
               "--max-model-len", "512", "--max-new-tokens", "192", "--mini-batch-size", "2",
               "--attention-backend", "fa4", "--kv-cache-layout", "paged", "--scheduler-policy", "paged",
               "--scheduler-num-device-pages", "32", "--block-length", "64", "--page-size", "64",
               "--parallel-decoding", decoder, "--apply-template"]
    def request(prompt, limit=192):
        return _request_json(url + "/v1/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": limit, "temperature": 0, "stream": False}, timeout=180)
    with (tmp_path / "server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_ready(url, process)
            prompts = ["What is 2 + 2? Answer with only the number.",
                       "Read this context carefully. " + "This is a geography question. " * 14
                       + "What is the capital of France? Answer with only the city name."]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                replies = list(pool.map(request, prompts))
            for (status, body), answer in zip(replies, ["4", "Paris"]):
                print(json.dumps(body, ensure_ascii=False), flush=True)
                assert status == 200, body
                assert body["choices"][0]["message"]["content"].strip() == answer
                assert body["choices"][0]["finish_reason"] == "stop"
            # New requests reuse freed slots/pages; token limit forces a length stop.
            status, body = request("Return the integers from 1 to 40 in order, separated by a comma and a space. Do not add any other text.")
            print(json.dumps(body, ensure_ascii=False), flush=True)
            assert status == 200, body
            assert [int(n) for n in re.findall(r"\d+", body["choices"][0]["message"]["content"])] == list(range(1, 41))
            assert body["choices"][0]["finish_reason"] == "stop"
            status, body = request("Count upwards from 1, separated by commas.", limit=8)
            print(json.dumps(body, ensure_ascii=False), flush=True)
            assert status == 200, body
            assert body["choices"][0]["finish_reason"] == "length"
            assert body["choices"][0]["message"]["content"] == "1, 2, 3,"
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            print("server log:", tmp_path / "server.log", flush=True)
