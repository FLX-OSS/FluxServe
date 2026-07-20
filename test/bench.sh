python -m fluxserve.cli bench serve \
  --model inclusionAI/LLaDA2.0-mini \
  --dataset ./data/humaneval.jsonl \
  --dataset-output-len 512 \
  --request-rate 1 \
  --max-concurrency 32 \
  --metric-percentiles 50,90,99