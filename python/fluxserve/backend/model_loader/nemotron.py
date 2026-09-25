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

"""Checkpoint loading and configuration normalization for Nemotron-Labs-Diffusion.

The shared loader path assumes a sharded checkpoint described by
``model.safetensors.index.json`` and a LLaDA-shaped MoE model. This checkpoint
is a single unsharded ``model.safetensors`` next to a LoRA adapter that must not
be loaded, so resolution and iteration live here instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn

from fluxserve.backend.models.nemotron_diffusion import (
    NemotronLabsDiffusionLLM,
    is_nemotron_diffusion_config,
    nemotron_head_dim,
    nemotron_weight_plan,
)
from fluxserve.backend.model_loader.weight_utils import get_model_name

logger = logging.getLogger(__name__)

WEIGHT_FILE = "model.safetensors"
GENERATION_CONFIG_FILE = "generation_config.json"

# Checkpoint architectural ceiling. Hardware/quality validation is recorded
# separately; request buffers must also respect the configured serving limit.
MAX_SUPPORTED_POSITIONS = 262144

# Selected through the existing --parallel-decoding flag rather than a new
# one; Nemotron overrides init_decoder, so the shared decoder factory never
# sees these values.
SELF_SPECULATION = "self_speculation"
NEMOTRON_DECODING_MODES = ("threshold", SELF_SPECULATION)


def resolve_nemotron_snapshot(model_config, *, local_files_only: bool = False) -> Path:
    """Locate the snapshot directory holding a single-file checkpoint.

    ``local_files_only`` makes resolution a cache lookup that raises rather
    than fetching. Tests use it: this checkpoint is 27 GB, and a helper that
    silently downloads it would turn a unit test into a long transfer on any
    machine whose cache happens to be cold.
    """
    model_name = get_model_name(model_config)
    local = Path(model_name).expanduser()
    if (local / WEIGHT_FILE).is_file():
        return local
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "huggingface_hub is required to resolve Nemotron model weights"
        ) from exc
    # Deliberately narrow: a bare "*.safetensors" pattern would also fetch
    # linear_spec_lora/adapter_model.safetensors, which is not a base weight.
    snapshot = Path(
        snapshot_download(
            repo_id=model_name,
            revision=getattr(model_config, "_commit_hash", None),
            allow_patterns=[WEIGHT_FILE, "*.json", "*.jinja"],
            local_files_only=local_files_only,
        )
    )
    if not (snapshot / WEIGHT_FILE).is_file():
        raise FileNotFoundError(
            f"Nemotron checkpoint '{model_name}' does not provide {WEIGHT_FILE}"
        )
    return snapshot


def read_safetensors_header(path: Path | str) -> dict[str, dict]:
    """Read a safetensors header without mapping or loading any tensor data."""
    with open(path, "rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def iter_nemotron_tensors(
    weight_file: Path | str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield one checkpoint tensor at a time.

    Reading the whole 27 GB file at once would need it resident in host memory
    before any GPU copy; ``safe_open`` keeps that bounded to one tensor.
    """
    from safetensors import safe_open

    with safe_open(str(weight_file), framework="pt") as handle:
        for name in handle.keys():
            yield name, handle.get_tensor(name)


def resolve_nemotron_eos_ids(model_config) -> tuple[int, ...]:
    """Collect EOS ids from ``generation_config.json``, falling back to config."""
    ids: list[int] = []
    try:
        # Cache-only: the caller is already loading this checkpoint, or is a
        # test. Neither wants an incidental download from an id lookup.
        snapshot = resolve_nemotron_snapshot(model_config, local_files_only=True)
        path = snapshot / GENERATION_CONFIG_FILE
        if path.is_file():
            with open(path, "r") as handle:
                raw = json.load(handle).get("eos_token_id")
            if isinstance(raw, int):
                ids = [raw]
            elif isinstance(raw, (list, tuple)):
                ids = [int(value) for value in raw]
    except (OSError, ValueError, RuntimeError):
        logger.debug("Nemotron generation_config.json unavailable", exc_info=True)
    if not ids:
        raw = getattr(model_config, "eos_token_id", None)
        if isinstance(raw, int):
            ids = [raw]
        elif isinstance(raw, (list, tuple)):
            ids = [int(value) for value in raw]
    deduplicated: list[int] = []
    for value in ids:
        if value not in deduplicated:
            deduplicated.append(int(value))
    if not deduplicated:
        raise ValueError("Nemotron checkpoint declares no EOS token id")
    return tuple(deduplicated)


def nemotron_decoding_ids(model_config) -> dict[str, object]:
    """Mask and EOS ids for this checkpoint.

    ``RunnerConfig`` defaults to LLaDA's ids, which are outside this model's
    131072-entry vocabulary; both the primary id and the tuple are set so the
    shared decoder factory keeps its existing ordering rule unchanged.
    """
    mask_id = getattr(model_config, "mask_token_id", None)
    vocab_size = int(model_config.vocab_size)
    if mask_id is None or not 0 <= int(mask_id) < vocab_size:
        raise ValueError(
            "Nemotron checkpoint must declare mask_token_id inside its "
            f"vocabulary, got {mask_id!r} for vocab_size={vocab_size}"
        )
    eos_ids = resolve_nemotron_eos_ids(model_config)
    for value in eos_ids:
        if not 0 <= value < vocab_size:
            raise ValueError(
                f"Nemotron EOS id {value} is outside vocab_size={vocab_size}"
            )
    return {
        "mask_id": int(mask_id),
        "eos_id": int(eos_ids[0]),
        "eos_ids": eos_ids,
    }


END_THINK_TOKEN = "</think>"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


def resolve_end_think_token_id(model_config) -> int | None:
    """The checkpoint's ``</think>`` id, or ``None`` when it declares none."""
    try:
        snapshot = resolve_nemotron_snapshot(model_config, local_files_only=True)
        path = snapshot / TOKENIZER_CONFIG_FILE
        if not path.is_file():
            return None
        with open(path) as handle:
            added = json.load(handle).get("added_tokens_decoder") or {}
    except (OSError, ValueError, RuntimeError):
        logger.debug("Nemotron tokenizer_config.json unavailable", exc_info=True)
        return None
    for token_id, entry in added.items():
        if isinstance(entry, dict) and entry.get("content") == END_THINK_TOKEN:
            return int(token_id)
    return None


def nemotron_block_length(model_config, requested: int | None = None) -> int:
    """Serving block length; the checkpoint's ``block_size`` is the default."""
    block_size = int(getattr(model_config, "block_size", 0) or 0)
    if requested is None:
        if block_size <= 0:
            raise ValueError("Nemotron checkpoint declares no block_size")
        return block_size
    requested = int(requested)
    if requested <= 0:
        raise ValueError(f"--block-length must be positive, got {requested}")
    return requested


def check_nemotron_context_limit(max_positions: int, model_config=None, *, serving_limit=None) -> None:
    """Check absolute positions, including provisional generation slots."""
    limit = min(MAX_SUPPORTED_POSITIONS, int(
        getattr(model_config, "max_position_embeddings", MAX_SUPPORTED_POSITIONS)
    ))
    if serving_limit is not None:
        limit = min(limit, int(serving_limit))
    if not 0 <= int(max_positions) <= limit:
        raise ValueError(
            "Nemotron-Labs-Diffusion support is currently limited to "
            f"{limit} total positions; requested {int(max_positions)}. "
            "This includes provisional block slots."
        )


LORA_SUBFOLDER = "linear_spec_lora"
LORA_WEIGHT_FILE = "adapter_model.safetensors"
LORA_CONFIG_FILE = "adapter_config.json"


class NemotronDraftAdapter:
    """The ``linear_spec_lora`` draft adapter, pre-fused into weight pairs.

    The adapter targets ``o_proj`` only and is toggled on for the drafting
    forward and off for verification, within a single iteration. Evaluating a
    low-rank update on every forward would put two extra matmuls per layer on
    the critical path for an adapter that never changes, so both weights are
    materialised once and the module's parameter is pointed at one of them.

    The cost is memory: one extra ``o_proj`` per layer, about 1.7 GB in
    bfloat16 for this checkpoint. The benefit is that toggling is a pointer
    swap and the drafting forward runs at exactly base-model speed.
    """

    def __init__(self, layers):
        self.layers = list(layers)
        self.enabled = False

    def apply(self, enabled: bool) -> None:
        if enabled == self.enabled:
            return
        for module, base, draft in self.layers:
            module.weight.data = draft if enabled else base
        self.enabled = bool(enabled)


def load_nemotron_lora(model, model_config, path=None):
    """Fuse the draft adapter into per-layer weight pairs, or return ``None``.

    Returns ``None`` when the checkpoint ships no adapter: self-speculation
    works without it at a lower acceptance length, so a missing adapter is a
    configuration, not an error.
    """
    from safetensors import safe_open

    if path is None:
        try:
            snapshot = resolve_nemotron_snapshot(model_config, local_files_only=True)
        except (OSError, ValueError, RuntimeError):
            logger.debug("Nemotron snapshot unavailable for LoRA", exc_info=True)
            return None
        path = snapshot / LORA_SUBFOLDER / LORA_WEIGHT_FILE
        config_path = snapshot / LORA_SUBFOLDER / LORA_CONFIG_FILE
    else:
        path = Path(path)
        config_path = path.parent / LORA_CONFIG_FILE
    if not Path(path).is_file():
        return None

    rank, alpha, targets = 0, 0.0, ("o_proj",)
    if Path(config_path).is_file():
        with open(config_path) as handle:
            raw = json.load(handle)
        rank = int(raw.get("r", 0) or 0)
        alpha = float(raw.get("lora_alpha", 0.0) or 0.0)
        targets = tuple(raw.get("target_modules") or targets)
    if set(targets) != {"o_proj"}:
        raise ValueError(
            "Nemotron draft adapter support covers o_proj only; this adapter "
            f"targets {sorted(targets)}"
        )

    layers = []
    with safe_open(str(path), framework="pt") as handle:
        names = set(handle.keys())
        for index, block in enumerate(model.model.layers):
            prefix = f"base_model.model.encoder.layers.{index}.self_attn.o_proj"
            a_name, b_name = f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"
            if a_name not in names or b_name not in names:
                raise ValueError(
                    f"Nemotron draft adapter is missing {a_name} / {b_name}"
                )
            lora_a = handle.get_tensor(a_name)
            lora_b = handle.get_tensor(b_name)
            if not rank:
                rank = int(lora_a.shape[0])
            if not alpha:
                alpha = float(rank)
            module = block.self_attn.o_proj
            base = module.weight.data
            # o_proj is row-parallel: its input dimension is sharded, so the
            # adapter's A matrix must be sliced the same way before fusing.
            shard_width = int(base.shape[1])
            if shard_width > lora_a.shape[1]:
                raise ValueError(
                    f"layer {index}: o_proj shard {shard_width} exceeds the "
                    f"adapter's input width {int(lora_a.shape[1])}"
                )
            shard_index = int(lora_a.shape[1]) // shard_width
            rank_id = 0 if shard_index <= 1 else _attention_tp_rank()
            start = rank_id * shard_width
            a_shard = lora_a[:, start : start + shard_width]
            delta = (alpha / rank) * (
                lora_b.to(torch.float32) @ a_shard.to(torch.float32)
            )
            draft = (base.to(torch.float32) + delta.to(base.device)).to(base.dtype)
            layers.append((module, base, draft))
    return NemotronDraftAdapter(layers)


def _attention_tp_rank() -> int:
    from fluxserve.backend.layers.dp_attention import get_attention_tp_rank

    try:
        return int(get_attention_tp_rank())
    except AssertionError:
        return 0


def apply_nemotron_runner_config(runner_config, model_config, args=None) -> None:
    """Point a ``RunnerConfig`` at this checkpoint's ids and options.

    ``RunnerConfig`` defaults to LLaDA's mask and EOS ids, which are outside
    this model's 131072-entry vocabulary. Both the primary id and the tuple are
    set so the shared decoder factory's ordering rule stays untouched.

    Thinking-budget settings are attached here rather than added to the shared
    schema; runners read them with ``getattr`` defaults, so a config built
    without them behaves exactly as before.
    """
    ids = nemotron_decoding_ids(model_config)
    runner_config.mask_id = ids["mask_id"]
    runner_config.eos_id = ids["eos_id"]
    runner_config.eos_ids = ids["eos_ids"]
    runner_config.max_thinking_tokens = getattr(args, "max_thinking_tokens", None)
    runner_config.end_think_token_id = getattr(args, "end_think_token_id", None)
    runner_config.temperature = getattr(args, "temperature", 0.0)
    runner_config.seed = getattr(args, "seed", None)
    runner_config.nemotron_prefill_chunk_size = getattr(args, "nemotron_prefill_chunk_size", 1024)


def normalize_nemotron_args(args, model_config) -> bool:
    """Validate and default Nemotron-only serving arguments.

    Returns ``True`` when the checkpoint is a Nemotron one, so callers can use
    it as the dispatch predicate. Shared by the server and the offline bench so
    the two cannot drift apart.
    """
    if not is_nemotron_diffusion_config(model_config):
        return False

    from fluxserve.backend.execution.nemotron_sampling import validate_sampling_params

    validate_sampling_params({"temperature": getattr(args, "temperature", 0.0),
                              "seed": getattr(args, "seed", None)})
    if int(getattr(args, "nemotron_prefill_chunk_size", 1024)) <= 0:
        raise ValueError("--nemotron-prefill-chunk-size must be positive")

    backend = getattr(args, "attention_backend", "sdpa")
    if backend not in ("sdpa", "fa4", "flashinfer"):
        raise ValueError(
            "Nemotron-Labs-Diffusion supports attention_backend 'sdpa' "
            f"(dense), 'fa4' or 'flashinfer' (paged); got {backend!r}."
        )
    if backend in ("fa4", "flashinfer"):
        if getattr(args, "kv_cache_layout", "dense") != "paged":
            raise ValueError(f"Nemotron {backend.upper()} serving requires --kv-cache-layout paged")
        if backend == "flashinfer":
            for name in ("flashinfer_cache_mode", "flashinfer_prefill_mode"):
                if getattr(args, name, "paged") != "paged":
                    raise ValueError(f"Nemotron FlashInfer requires --{name.replace('_', '-')} paged")
    elif hasattr(args, "kv_cache_layout"):
        # `--kv-cache-layout` defaults to paged, but the dense runner owns a
        # preallocated buffer and ignores the paged manager entirely. Normalize
        # it so the resolved configuration describes what actually runs instead
        # of advertising a cache the model never touches.
        args.kv_cache_layout = "dense"
    if getattr(args, "use_prefill_cuda_graph", False):
        raise ValueError(
            "Nemotron prefill is a single variable-length causal forward and "
            "is not captured; use --use-decode-cuda-graph."
        )
    wants_graph = getattr(args, "use_cuda_graph", False) or getattr(
        args, "use_decode_cuda_graph", False
    )
    if wants_graph and backend not in ("fa4", "flashinfer"):
        raise ValueError(
            "Nemotron decode CUDA graphs exist only on the paged FA4 or "
            "FlashInfer paths; use --attention-backend fa4 or flashinfer with "
            "--kv-cache-layout paged."
        )
    if getattr(args, "scheduler_policy", "") == "paged" and backend not in ("fa4", "flashinfer"):
        raise ValueError(
            "Nemotron continuously scheduled paged serving requires "
            "--attention-backend fa4 or flashinfer with --kv-cache-layout paged."
        )
    decoding = getattr(args, "parallel_decoding", "threshold")
    if decoding not in NEMOTRON_DECODING_MODES:
        raise ValueError(
            "Nemotron supports --parallel-decoding "
            f"{' or '.join(sorted(NEMOTRON_DECODING_MODES))}; got {decoding!r}."
        )
    if (
        decoding == SELF_SPECULATION
        and getattr(args, "scheduler_policy", "") == "paged"
        and backend not in ("fa4", "flashinfer")
    ):
        raise ValueError(
            "Continuously scheduled self-speculation requires paged FA4 or FlashInfer; "
            "the dense runner serves one request at a time."
        )

    block_length = nemotron_block_length(
        model_config, getattr(args, "block_length", None)
    )
    model_block = int(getattr(model_config, "block_size", 0) or 0)
    if model_block and block_length % model_block:
        raise ValueError(
            f"--block-length ({block_length}) must be a multiple of the "
            f"checkpoint's block_size ({model_block})"
        )
    args.block_length = block_length

    max_length = int(getattr(args, "max_model_len", 0) or 0)
    if max_length:
        check_nemotron_context_limit(max_length, model_config)

    budget = getattr(args, "max_thinking_tokens", None)
    if budget is not None:
        if int(budget) < 0:
            raise ValueError(
                f"--max-thinking-tokens must be non-negative, got {budget}"
            )
        marker = getattr(args, "end_think_token_id", None)
        if marker is None:
            marker = resolve_end_think_token_id(model_config)
        if marker is None:
            raise ValueError(
                "--max-thinking-tokens needs an end-of-thinking token; this "
                "checkpoint declares no '</think>', so pass "
                "--end-think-token-id explicitly"
            )
        if not 0 <= int(marker) < int(model_config.vocab_size):
            raise ValueError(
                f"--end-think-token-id {marker} is outside vocab_size="
                f"{int(model_config.vocab_size)}"
            )
        args.end_think_token_id = int(marker)
    elif getattr(args, "end_think_token_id", None) is not None:
        raise ValueError(
            "--end-think-token-id has no effect without --max-thinking-tokens"
        )
    return True


class NemotronModelLoader:
    """Load the single-file Nemotron checkpoint into the dense model."""

    def load_model(self, *, model_config, device: str, quant_config=None) -> nn.Module:
        if quant_config is not None:
            raise ValueError(
                "Nemotron-Labs-Diffusion quantization is not supported yet"
            )
        if not is_nemotron_diffusion_config(model_config):
            raise ValueError(
                "NemotronModelLoader received a non-Nemotron config: "
                f"architectures={getattr(model_config, 'architectures', None)!r}"
            )
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device(device):
                model = NemotronLabsDiffusionLLM(model_config).eval()
        finally:
            torch.set_default_dtype(old_dtype)

        snapshot = resolve_nemotron_snapshot(model_config)
        weight_file = snapshot / WEIGHT_FILE
        plan = nemotron_weight_plan(model_config)

        with torch.inference_mode():
            consumed, unexpected = model.load_weights(
                iter_nemotron_tensors(weight_file)
            )

        if unexpected:
            preview = ", ".join(sorted(unexpected)[:20])
            raise ValueError(
                f"Nemotron checkpoint contains {len(unexpected)} unmapped "
                f"tensors: {preview}"
            )
        missing = sorted(set(plan) - consumed)
        if missing:
            preview = ", ".join(missing[:20])
            raise ValueError(
                f"Nemotron checkpoint is missing {len(missing)} required "
                f"tensors: {preview}"
            )
        logger.info(
            "Nemotron-Labs-Diffusion loaded: %d checkpoint tensors, "
            "%d layers, head_dim=%d, 0 unmatched keys",
            len(consumed),
            int(model_config.num_hidden_layers),
            nemotron_head_dim(model_config),
        )
        return model.eval()


__all__ = [
    "MAX_SUPPORTED_POSITIONS",
    "NEMOTRON_DECODING_MODES",
    "SELF_SPECULATION",
    "NemotronDraftAdapter",
    "NemotronModelLoader",
    "load_nemotron_lora",
    "apply_nemotron_runner_config",
    "resolve_end_think_token_id",
    "normalize_nemotron_args",
    "check_nemotron_context_limit",
    "iter_nemotron_tensors",
    "nemotron_block_length",
    "nemotron_decoding_ids",
    "read_safetensors_header",
    "resolve_nemotron_eos_ids",
    "resolve_nemotron_snapshot",
]
