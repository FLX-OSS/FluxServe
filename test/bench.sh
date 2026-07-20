python -m fluxserve.cli bench_offline \
  --model inclusionAI/LLaDA2.0-mini \
  --dataset ./data/humaneval.jsonl \
  --tp-size 4 \
  --dp-size 1 \
  --ep-size 4 \
  --batch-size 4 \
  --gen-len 512 \
  --block-length 64


# python -m fluxserve.cli bench serve \
#   --model inclusionAI/LLaDA2.0-mini \
#   --dataset ./data/humaneval.jsonl \
#   --dataset-output-len 512 \
#   --request-rate 1 \
#   --max-concurrency 32 \
#   --metric-percentiles 50,90,99