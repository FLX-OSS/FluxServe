#!/usr/bin/env bash
set -euo pipefail

# Keep this script identical between local development and GitHub Actions.
# Run it after activating the virtual environment containing FluxServe.
# It intentionally runs only tests that do not require a CUDA device.

python -m pytest -q \
  test/test_scheduler_defaults.py \
  test/test_quantization_removed.py \
  test/test_prompt_rendering.py
