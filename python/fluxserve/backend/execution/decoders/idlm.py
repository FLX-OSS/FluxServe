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

"""
    I-DLM Introspective Strided Decoding (ISD).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from fluxserve.backend.execution.decoders.common import normalize_eos_ids


def validate_sampling_params(params):
    """Request sampling parameters I-DLM can honour."""
    temperature = params.get("temperature", 0.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a finite non-negative number")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite non-negative number")
    seed = params.get("seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
    ):
        raise ValueError("seed must be an integer in [0, 2**63)")
    top_k = params.get("top_k", -1)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < -1:
        raise ValueError("top_k must be -1 (disabled), 0 (disabled) or a positive integer")
    top_p = params.get("top_p", 1.0)
    if (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not math.isfinite(top_p)
        or not 0.0 < top_p <= 1.0
    ):
        raise ValueError("top_p must be in (0, 1]")
    for name in ("frequency_penalty", "presence_penalty"):
        if params.get(name, 0.0) != 0.0:
            raise ValueError(f"I-DLM does not support {name}")


def suppress_token(logits: torch.Tensor, token_id: int | None) -> torch.Tensor:
    """Return ``logits`` with ``token_id`` made unreachable (a copy)."""
    if token_id is None:
        return logits
    logits = logits.clone()
    logits[..., int(token_id)] = -torch.inf
    return logits


def inverse_cdf(weights: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    """Draw an index from each row of non-negative ``weights`` ``[..., V]``.

    The draw is a pure function of ``uniform`` ``[...]`` in ``[0, 1)``, so it can
    run for many rows at once while each row's randomness still comes from its
    own request generator. Rows need not be normalized.
    """
    cdf = weights.cumsum(dim=-1)
    target = (uniform * cdf[..., -1]).unsqueeze(-1)
    index = torch.searchsorted(cdf, target.contiguous(), right=True).squeeze(-1)
    return index.clamp_(max=weights.shape[-1] - 1)


def sample_top_k_top_p(scaled: torch.Tensor, top_k, top_p, uniform: torch.Tensor,
                       limit: int | None = None) -> torch.Tensor:
    """One token per row of ``scaled`` ``[R, V]`` (float32 logits over T).

    Joint top-k/top-p, as FlashInfer's ``filter_apply_order="joint"`` that the
    reference sampler uses: a token survives only inside both the top-k and the
    top-p nucleus of the unfiltered distribution. Both are prefixes of the
    descending order, so the most likely token always survives. Parameters are
    per row: ``top_k`` is ``V`` and ``top_p`` is ``inf`` where disabled.
    ``limit`` bounds every row's top-k; below ``V`` only that many candidates
    are ranked, with probabilities still over the full vocabulary.
    """
    vocab = int(scaled.shape[-1])
    limit = vocab if limit is None else max(1, min(int(limit), vocab))
    if limit < vocab:
        values, order = scaled.topk(limit, dim=-1)
    else:
        values, order = scaled.sort(dim=-1, descending=True)
    probs = (values - torch.logsumexp(scaled, dim=-1, keepdim=True)).exp()
    device = scaled.device

    def column(value, dtype):
        # A Python number becomes a fill, not a host-to-device copy.
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype).reshape(-1, 1)
        return torch.full((1, 1), value, device=device, dtype=dtype)

    top_k, top_p = column(top_k, torch.long), column(top_p, probs.dtype)
    rank = torch.arange(limit, device=device)
    # Mass strictly before each token: the token that crosses top_p is kept.
    keep = (rank < top_k) & ((probs.cumsum(dim=-1) - probs) < top_p)
    choice = inverse_cdf(probs * keep, uniform)
    return order.gather(-1, choice.unsqueeze(-1)).squeeze(-1)


def _staged(values, dtype, device) -> torch.Tensor:
    """Host values on ``device`` without waiting for queued GPU work."""
    pinned = torch.device(device).type == "cuda"
    return torch.tensor(values, dtype=dtype, pin_memory=pinned).to(device, non_blocking=pinned)


@dataclass
class ISDSampling:
    """Per-request sampling state; its generator is the only source of randomness.

    ``suppress_id`` is the MASK token, which the served stream must never
    contain. The reference does not exclude it; for a trained checkpoint the
    exclusion is a no-op.
    """

    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    seed: int = 0
    suppress_id: int | None = None
    _generators: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        validate_sampling_params({
            "temperature": self.temperature, "seed": self.seed,
            "top_k": self.top_k, "top_p": self.top_p,
        })

    @property
    def greedy(self) -> bool:
        return self.temperature == 0

    def generator(self, device):
        key = str(device)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(self.seed)
        return self._generators[key]

    def scaled(self, logits: torch.Tensor) -> torch.Tensor:
        """Float32 logits over temperature, with the MASK token excluded."""
        logits = suppress_token(logits.float(), self.suppress_id)
        return logits if self.greedy or self.temperature == 1.0 else logits / self.temperature

    def bounds(self, vocab: int) -> tuple[int, float]:
        """``(top_k, top_p)`` for ``sample_top_k_top_p``: ``V`` and ``inf`` when off."""
        top_k = self.top_k if 0 < self.top_k < vocab else vocab
        return top_k, self.top_p if self.top_p < 1.0 else math.inf

    def categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample one token per row: argmax at T=0, else joint top-k/top-p.

        The prefill seed comes from here, and ``ISDDecoder`` samples bonus
        tokens with the same ``sample_top_k_top_p``.
        """
        scaled = self.scaled(logits)
        if self.greedy:
            return scaled.argmax(dim=-1)
        flat = scaled.reshape(-1, scaled.shape[-1])
        top_k, top_p = self.bounds(int(flat.shape[-1]))
        uniform = torch.rand(
            flat.shape[0], device=flat.device, generator=self.generator(flat.device)
        )
        return sample_top_k_top_p(flat, top_k, top_p, uniform, limit=top_k).reshape(
            logits.shape[:-1]
        )


def make_sampling(config=None, params=None) -> ISDSampling:
    values = {
        "temperature": getattr(config, "temperature", 0.0),
        "seed": getattr(config, "seed", 0),
        "top_k": getattr(config, "top_k", -1),
        "top_p": getattr(config, "top_p", 1.0),
    }
    values.update({
        key: value for key, value in (params or {}).items()
        if key in values or key in ("frequency_penalty", "presence_penalty")
    })
    validate_sampling_params(values)
    top_k = values["top_k"]
    return ISDSampling(
        temperature=float(values["temperature"]),
        top_k=-1 if top_k in (None, 0) else int(top_k),
        top_p=float(values["top_p"]),
        # Online input processing assigns unseeded requests a seed before
        # rank broadcasting; offline the default is explicit and identical.
        seed=values["seed"] or 0,
        suppress_id=getattr(config, "mask_id", None),
    )


@dataclass
class ISDOutcome:
    """What one ISD step decided for one request.

    ``emitted`` excludes the anchor; the caller prepends the anchor when it has
    not been emitted yet (only the prefill seed). ``advance`` is how far the
    committed KV prefix moves: the anchor plus the accepted specs.
    """

    emitted: list[int]
    advance: int
    next_anchor: int
    next_specs: list[int]
    next_draft_logits: torch.Tensor | None
    accepted: int
    rejected: bool


class ISDDecoder:
    """Resolve one causal forward over ``[anchor, specs..., MASK...]`` per row.

    With ``N = gen_block_size`` the forward is ``W = 2N - 1`` wide. Logits at
    block offset ``j`` predict the token at offset ``j + 1``. With ``k`` specs,
    offsets ``0..k-1`` verify them left to right, offset ``k`` is the bonus
    token, and offsets ``k+1..k+N-1`` draft the next step; the rest pad a cold
    start. This is ``IDLMBlockN.run`` from the reference, restated over
    positions so that rollback is just a shorter prefix advance.

    * Greedy: a spec is accepted iff it equals the argmax, and the correction
      is the argmax. The output is then exactly greedy AR decoding.
    * Sampled: drafts are argmax, as in the reference. A spec ``s`` is accepted
      iff ``u < p(s) / q(s)`` with ``p = softmax(clean / T)`` and
      ``q = softmax(draft / T)`` over the full vocabulary. A rejection draws
      the correction from ``max(0, p - q)``, falling back to ``p``. The bonus
      token uses joint top-k/top-p.
    """

    def __init__(self, *, gen_block_size: int, mask_id: int, eos_ids):
        if int(gen_block_size) < 1:
            raise ValueError(f"gen_block_size must be >= 1, got {gen_block_size!r}")
        self.gen_block_size = int(gen_block_size)
        self.num_specs = self.gen_block_size - 1
        self.width = 2 * self.gen_block_size - 1
        self.mask_id = int(mask_id)
        self.eos_ids = normalize_eos_ids(eos_ids)
        self.eos_id = self.eos_ids[0]

    def block(self, anchor: int, specs) -> list[int]:
        specs = list(specs)
        if len(specs) not in (0, self.num_specs):
            raise ValueError(
                f"an ISD block carries 0 or {self.num_specs} specs, got {len(specs)}"
            )
        return [int(anchor), *map(int, specs)] + [self.mask_id] * (self.width - 1 - len(specs))

    # -- resolution --------------------------------------------------------

    def resolve(self, logits: torch.Tensor, specs, draft_logits, samplings) -> list[ISDOutcome]:
        """Decide every row of one step with a single device-to-host copy.

        ``logits`` is ``[B, W, V]``; ``specs[b]`` is the row's spec list (empty
        for a cold start); ``draft_logits[b]`` is ``[k, V]`` scaled draft logits
        for a sampled verify row and ``None`` otherwise.
        """
        batch, width = int(logits.shape[0]), int(logits.shape[1])
        if width != self.width:
            raise ValueError(f"ISD logits are {width} wide, expected {self.width}")
        if not len(specs) == len(draft_logits) == len(samplings) == batch:
            raise ValueError("specs, draft_logits and samplings need one entry per row")
        greedy = [b for b in range(batch) if samplings[b].greedy]
        sampled = [b for b in range(batch) if not samplings[b].greedy]

        parts: list[torch.Tensor] = []
        if greedy:
            rows = logits if len(greedy) == batch else logits.index_select(
                0, _staged(greedy, torch.long, logits.device)
            )
            # Greedy rows share one argmax; the MASK token is excluded first.
            parts.append(suppress_token(rows, self.mask_id).argmax(dim=-1).flatten())
        next_drafts: list[torch.Tensor] = []
        if sampled:
            sampled_parts, next_drafts = self._sampled_rows(
                logits, sampled, specs, draft_logits, samplings
            )
            parts += sampled_parts
        values = torch.cat(parts).tolist() if parts else []

        outcomes: list[ISDOutcome | None] = [None] * batch
        cursor = 0

        def take(size):
            nonlocal cursor
            cursor += size
            return values[cursor - size : cursor]

        for b in greedy:
            outcomes[b] = self._greedy_outcome(take(width), list(specs[b]))
        if sampled:
            k = self.num_specs
            verified = sum(1 for b in sampled if specs[b])
            accepts, corrections = take(verified * k), take(verified * k)
            bonuses, drafts = take(len(sampled)), take(len(sampled) * k)
            row = 0
            for index, b in enumerate(sampled):
                pack = []
                if specs[b]:
                    pack = accepts[row * k : (row + 1) * k] + corrections[row * k : (row + 1) * k]
                    row += 1
                pack += [bonuses[index], *drafts[index * k : (index + 1) * k]]
                outcomes[b] = self._sampled_outcome(pack, list(specs[b]), next_drafts[index])
        return outcomes

    def _greedy_outcome(self, argmax: list[int], specs: list[int]) -> ISDOutcome:
        k = len(specs)
        for index, spec in enumerate(specs):
            if int(argmax[index]) != int(spec):
                correction = int(argmax[index])
                return ISDOutcome(
                    emitted=[*specs[:index], correction], advance=1 + index,
                    next_anchor=correction, next_specs=[], next_draft_logits=None,
                    accepted=index, rejected=True,
                )
        clean = int(argmax[k])
        return ISDOutcome(
            emitted=[*specs, clean], advance=1 + k,
            next_anchor=clean,
            next_specs=[int(value) for value in argmax[k + 1 : k + self.gen_block_size]],
            next_draft_logits=None, accepted=k, rejected=False,
        )

    def _sampled_rows(self, logits, rows, specs, draft_logits, samplings):
        """Device work for every sampled row of one step at once.

        Returns the flat device parts -- ``accept`` and ``correction`` (``k``
        each, verify rows only), then ``bonus`` and ``drafts`` (``N - 1`` each,
        every row) -- and each row's next draft logits ``[N - 1, V]``.

        The softmaxes, the residual and the top-k/top-p ranking are batched.
        Only the uniforms are drawn row by row, from each request's own
        generator, and every draw is an inverse-CDF lookup of one of them, so a
        row's tokens depend on its seed and not on the rest of the batch. Host
        values are staged without a sync, so the step still syncs once.
        """
        device = logits.device
        vocab = int(logits.shape[-1])
        k = self.num_specs
        count = len(rows)
        verify = [index for index, b in enumerate(rows) if specs[b]]
        for index in verify:
            stored = draft_logits[rows[index]]
            if stored is None or tuple(stored.shape) != (k, vocab):
                raise RuntimeError("a sampled verify row needs its drafts' logits")

        # Per row: k accept draws, k correction draws, one bonus draw. A cold
        # row uses only the last, so every request consumes the same stream.
        uniform = torch.stack([
            torch.rand(2 * k + 1, device=device, generator=samplings[b].generator(device))
            for b in rows
        ])
        bounds = [samplings[b].bounds(vocab) for b in rows]
        temperature = _staged([samplings[b].temperature for b in rows], torch.float32, device)
        top_k = _staged([value for value, _ in bounds], torch.long, device)
        top_p = _staged([value for _, value in bounds], torch.float32, device)
        offsets = _staged([k if specs[b] else 0 for b in rows], torch.long, device)

        selected = logits if count == logits.shape[0] else logits.index_select(
            0, _staged(rows, torch.long, device)
        )
        scaled = selected.to(torch.float32, copy=True)
        scaled[..., self.mask_id] = -torch.inf
        scaled /= temperature[:, None, None]

        parts: list[torch.Tensor] = []
        if verify:
            verify_index = (
                None if len(verify) == count else _staged(verify, torch.long, device)
            )

            def pick(tensor):
                return tensor if verify_index is None else tensor.index_select(0, verify_index)

            p = torch.softmax(pick(scaled)[:, :k], dim=-1)
            q = torch.softmax(torch.stack([
                draft_logits[rows[index]].to(device=device, dtype=torch.float32)
                for index in verify
            ]), dim=-1)
            spec_ids = _staged([list(specs[rows[index]]) for index in verify], torch.long, device)
            p_spec = p.gather(-1, spec_ids.unsqueeze(-1)).squeeze(-1)
            q_spec = q.gather(-1, spec_ids.unsqueeze(-1)).squeeze(-1)
            ratio = torch.where(q_spec > 0, p_spec / q_spec, torch.zeros_like(p_spec))
            draws = pick(uniform)
            accept = (ratio >= 1.0) | (draws[:, :k] < ratio)
            residual = torch.clamp(p - q, min=0)
            residual = torch.where(residual.sum(dim=-1, keepdim=True) > 0, residual, p)
            correction = inverse_cdf(residual, draws[:, k : 2 * k])
            parts += [accept.long().flatten(), correction.flatten()]

        arange = torch.arange(count, device=device)
        bonus = sample_top_k_top_p(
            scaled[arange, offsets], top_k, top_p, uniform[:, 2 * k],
            limit=max(value for value, _ in bounds),
        )
        positions = offsets[:, None] + 1 + torch.arange(k, device=device)
        # Gathered, so the kept q rows do not pin the whole [S, W, V] buffer.
        next_logits = scaled[arange[:, None], positions]
        parts += [bonus.long(), next_logits.argmax(dim=-1).flatten()]
        return parts, list(next_logits.unbind(0))

    def _sampled_outcome(self, pack, specs, drafts_logits) -> ISDOutcome:
        k = len(specs)
        accept, correction = pack[:k], pack[k : 2 * k]
        for index in range(k):
            if not accept[index]:
                corrected = int(correction[index])
                return ISDOutcome(
                    emitted=[*specs[:index], corrected], advance=1 + index,
                    next_anchor=corrected, next_specs=[], next_draft_logits=None,
                    accepted=index, rejected=True,
                )
        clean = int(pack[2 * k])
        return ISDOutcome(
            emitted=[*specs, clean], advance=1 + k, next_anchor=clean,
            next_specs=[int(value) for value in pack[2 * k + 1 :]],
            next_draft_logits=drafts_logits if self.num_specs else None,
            accepted=k, rejected=False,
        )

    # -- EOS ---------------------------------------------------------------

    def first_eos_index(self, tokens: torch.Tensor) -> torch.Tensor:
        """Index of the first EOS per row, or ``tokens.shape[-1]`` if none."""
        hit = torch.zeros_like(tokens, dtype=torch.bool)
        for eos_id in self.eos_ids:
            hit |= tokens == eos_id
        length = tokens.shape[-1]
        indices = torch.arange(length, device=tokens.device).expand_as(tokens)
        return torch.where(hit, indices, torch.full_like(indices, length)).min(dim=-1).values


__all__ = [
    "ISDDecoder",
    "ISDOutcome",
    "ISDSampling",
    "inverse_cdf",
    "make_sampling",
    "sample_top_k_top_p",
    "suppress_token",
    "validate_sampling_params",
]
