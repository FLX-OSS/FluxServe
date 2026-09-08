# Quickstart

Launch LLaDA2.0-mini and send a chat completion request. Complete the [Docker installation](getting_started.md) first; the commands below run inside that environment.

## Launch the server

```bash
CUDA_VISIBLE_DEVICES=0 fluxserve serve \
  --model inclusionAI/LLaDA2.0-mini \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 1 --dp-size 1 --ep-size 1 \
  --attention-backend flashinfer \
  --kv-cache-layout paged \
  --scheduler-policy paged
```

The checkpoint is downloaded on first use. Wait for model initialization before sending a request. Keep this process running.

## Check readiness

In another shell in the same environment:

```bash
curl -fsS http://127.0.0.1:8000/health
```

## Send a request

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "inclusionAI/LLaDA2.0-mini",
    "messages": [{"role": "user", "content": "Explain diffusion language models in a few sentences."}],
    "max_tokens": 128,
    "temperature": 0,
    "stream": false
  }'
```

Read the generated text in the response's `choices` field. The exact output varies with the checkpoint and decoding configuration.

## Go further

- [Tune a single-GPU deployment](../serving/llada2-mini.md).
- [Serve LLaDA2.0-flash on four GPUs](../serving/llada2-flash.md).
- [Measure throughput and latency](benchmark.md).
