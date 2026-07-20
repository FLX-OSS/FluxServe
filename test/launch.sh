python -m fluxserve.cli serve \
  --model inclusionAI/LLaDA2.0-mini \
  --host 0.0.0.0 \
  --port 8000 \
  --tp-size 4 \
  --ep-size 4 \
  --dp-size 1