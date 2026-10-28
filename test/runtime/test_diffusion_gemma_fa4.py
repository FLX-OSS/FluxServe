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

"""Diffusion-Gemma FA4 sliding layers without a GPU.

``ReferenceFA4`` implements ``flash_attn_varlen_func``'s paged contract (cu_seqlens_q,
seqused_k, page_table, bottom-right windows, window_left_anchor) in fp32 and FlashInfer's
append is a torch scatter, so these tests check FluxServe's metadata and window mapping
against dense attention over token positions; the kernels are tested in flash-attn-dllm.
"""

import math
from types import SimpleNamespace

import pytest
import torch

import fluxserve.backend.layers.attention.diffusion_gemma_flashinfer as gemma_attention
from fluxserve.backend.execution.flashinfer_cuda_graph_runner import (
    FlashInferCudaGraphRunner,
)
from fluxserve.backend.execution.runners.diffusion_gemma.fa4 import (
    DiffusionGemmaFA4Runner,
)
from fluxserve.backend.layers.attention.diffusion_gemma_flashinfer import (
    DiffusionGemmaLayerGeometry,
    DiffusionGemmaPagedAttention,
    DiffusionGemmaPagedKVCache,
)
from fluxserve.backend.layers.attention.diffusion_gemma_window import (
    fa4_window_kwargs,
    sliding_window_mask,
)

W = 16  # sliding window: small, so short prefixes reach W - CANVAS + 1
CANVAS = 8
PAGE = 128
H, HKV, D = 4, 2, 16
SCALE = 0.5


def torch_append(k, v, batch_indices, positions, paged_kv_cache, kv_indices, kv_indptr,
                 last_page_len, kv_layout="NHD"):
    k_cache, v_cache = paged_kv_cache
    page_size = k_cache.shape[1]
    for token in range(k.shape[0]):
        row, pos = int(batch_indices[token]), int(positions[token])
        page = int(kv_indices[int(kv_indptr[row]) + pos // page_size])
        k_cache[page, pos % page_size] = k[token]
        v_cache[page, pos % page_size] = v[token]


class ReferenceFA4:
    """flash_attn_varlen_func on a paged cache, in fp32; records every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, q, k_cache, v_cache, *, cu_seqlens_q, cu_seqlens_k, seqused_k,
                 max_seqlen_q, max_seqlen_k, page_table, softmax_scale, return_lse,
                 causal=False, window_size=(None, None), qv=None, **kwargs):
        anchor = kwargs.pop("window_left_anchor", "query")
        assert not kwargs and cu_seqlens_k is None and not return_lse
        self.calls.append(dict(causal=causal, window_size=window_size, anchor=anchor,
                               anchor_passed=anchor != "query", max_seqlen_k=max_seqlen_k,
                               kv_lens=seqused_k.tolist(), page_table=page_table.clone(),
                               mla=qv is not None, softmax_scale=softmax_scale))
        left, right = window_size
        if causal:
            right = 0
        page = k_cache.shape[1]
        out = torch.zeros_like(q)
        for b in range(seqused_k.numel()):
            q0, q1 = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
            sq, sk = q1 - q0, int(seqused_k[b])
            assert sq <= max_seqlen_q and sk <= max_seqlen_k
            pages = page_table[b, : math.ceil(sk / page)].long()
            hkv, hq = k_cache.shape[2], q.shape[1]
            k = k_cache[pages].reshape(-1, hkv, k_cache.shape[-1])[:sk].float().repeat_interleave(hq // hkv, 1)
            v = v_cache[pages].reshape(-1, hkv, v_cache.shape[-1])[:sk].float().repeat_interleave(hq // hkv, 1)
            rows, cols = torch.arange(sq)[:, None], torch.arange(sk)[None, :]
            offset = sk - sq  # bottom-right alignment
            visible = torch.ones(sq, sk, dtype=torch.bool)
            if left is not None:
                first = rows if anchor == "query" else torch.zeros_like(rows)
                visible &= cols >= first + offset - left
            if right is not None:
                visible &= cols <= rows + offset + right
            scores = torch.einsum("qhd,khd->hqk", q[q0:q1].float(), k)
            if qv is not None:  # MLA: S = q . k + qv . v
                scores = scores + torch.einsum("qhd,khd->hqk", qv[q0:q1].float(), v)
            scores = scores * softmax_scale
            probs = scores.masked_fill(~visible, float("-inf")).softmax(-1)
            o = torch.einsum("hqk,khd->qhd", probs, v)
            if qv is None:
                out[q0:q1] = o.to(q.dtype)
            else:
                out = out if out.shape[-1] == v.shape[-1] else torch.zeros(
                    q.shape[0], q.shape[1], v.shape[-1], dtype=q.dtype)
                out[q0:q1] = o.to(q.dtype)
        return out


def dense_attention(q, k, v, *, q_offset, phase, mode, window_left=None):
    """q (q_len, H, D) at positions q_offset.., k/v (kv_len, HKV, D) at positions 0.."""
    q_pos, k_pos = torch.arange(q.shape[0]) + q_offset, torch.arange(k.shape[0])
    visible = sliding_window_mask(q_pos, k_pos, phase=phase, sliding_window=W, mode=mode,
                                  window_left=window_left, q_start=q_offset)
    if phase != "denoise":
        visible &= k_pos[None, :] <= q_pos[:, None]
    k, v = k.repeat_interleave(H // HKV, 1), v.repeat_interleave(H // HKV, 1)
    scores = torch.einsum("qhd,khd->hqk", q, k) * SCALE
    probs = scores.masked_fill(~visible, float("-inf")).softmax(-1)
    return torch.einsum("hqk,khd->qhd", probs, v)


def _padded(parts, heads, length):
    out = torch.zeros(len(parts), heads, length, D)
    for row, part in enumerate(parts):
        out[row, :, : part.shape[0]] = part.transpose(0, 1)
    return out


def test_fa4_window_kwargs():
    assert fa4_window_kwargs(phase="prefill", sliding_window=1024) == dict(
        causal=True, window_size=(1023, None), window_left_anchor="query")
    assert fa4_window_kwargs(phase="commit", sliding_window=1024, mode="per_row") == dict(
        causal=True, window_size=(1023, None), window_left_anchor="query")
    assert fa4_window_kwargs(phase="denoise", sliding_window=1024) == dict(
        causal=False, window_size=(1023, 1023), window_left_anchor="query")
    assert fa4_window_kwargs(phase="denoise", sliding_window=1024, mode="per_row") == dict(
        causal=False, window_size=(1023, 1023), window_left_anchor="first_query")
    assert fa4_window_kwargs(phase="denoise", sliding_window=1024, mode="per_row",
                             window_left=1024)["window_size"] == (1024, 1023)


def test_sliding_window_modes_differ_only_for_long_prefixes():
    for prefix in range(0, 3 * W):
        q, k = torch.arange(prefix, prefix + CANVAS), torch.arange(prefix + CANVAS)
        per_query = sliding_window_mask(q, k, phase="denoise", sliding_window=W)
        per_row = sliding_window_mask(q, k, phase="denoise", sliding_window=W, mode="per_row",
                                      q_start=prefix)
        assert torch.equal(per_query, (q[:, None] - k[None, :]).abs() < W)
        assert torch.equal(per_row, (k[None, :] >= prefix - (W - 1)).expand(CANVAS, -1))
        assert torch.equal(per_query, per_row) == (prefix < W - CANVAS + 1)
        commit = sliding_window_mask(q, k, phase="commit", sliding_window=W, mode="per_row",
                                     q_start=prefix)
        assert torch.equal(commit, per_query)  # the mode only changes denoise


@pytest.mark.parametrize("mode", ["per_query", "per_row"])
def test_fa4_sliding_layer_matches_dense_attention(monkeypatch, mode):
    monkeypatch.setattr(gemma_attention, "require_diffusion_gemma_flashinfer",
                        lambda: (None, None, None, torch_append))
    prompts = (3, 20, 9, 140)  # around W - CANVAS + 1 = 9 and across a page
    seq_ids = (1, 3, 2, 4)  # non-trivial page-table rows
    totals = tuple(p + CANVAS for p in prompts)
    gen = torch.Generator().manual_seed(0)
    Q = [torch.randn(t, H, D, generator=gen) for t in totals]
    K = [torch.randn(t, HKV, D, generator=gen) for t in totals]
    V = [torch.randn(t, HKV, D, generator=gen) for t in totals]
    cache = DiffusionGemmaPagedKVCache(
        layer_geometries=[DiffusionGemmaLayerGeometry(HKV, D)], max_length=2 * PAGE,
        page_size=PAGE, dtype=torch.float32, device="cpu", batch_size=5,
        sliding_backend="fa4", sliding_window_mode=mode)
    kernel = ReferenceFA4()
    attention = DiffusionGemmaPagedAttention(layer_id=0, num_heads=H, num_kv_heads=HKV,
                                             head_dim=D, scale=SCALE, sliding_window=W,
                                             fa4_kernel=kernel)

    def run(phase, q_offsets, q_lens, max_q_len):
        metadata = cache.build_metadata(
            phase=phase, seq_ids=seq_ids, q_offsets=q_offsets, q_lens=q_lens,
            kv_lens=tuple(o + n for o, n in zip(q_offsets, q_lens)), max_q_len=max_q_len)
        parts = [slice(o, o + n) for o, n in zip(q_offsets, q_lens)]
        out = attention.forward(
            _padded([x[s] for x, s in zip(Q, parts)], H, max_q_len),
            _padded([x[s] for x, s in zip(K, parts)], HKV, max_q_len),
            _padded([x[s] for x, s in zip(V, parts)], HKV, max_q_len),
            cache=cache, metadata=metadata)
        for row, (offset, n) in enumerate(zip(q_offsets, q_lens)):
            expected = dense_attention(Q[row][offset:offset + n], K[row][:offset + n],
                                       V[row][:offset + n], q_offset=offset, phase=phase,
                                       mode=mode)
            torch.testing.assert_close(out[row, :, :n].transpose(0, 1), expected,
                                       atol=1e-5, rtol=1e-5)
        return kernel.calls[-1]

    prefill = run("prefill", (0,) * 4, prompts, max(prompts))
    denoise = run("denoise", prompts, (CANVAS,) * 4, CANVAS)
    commit = run("commit", prompts, (CANVAS,) * 4, CANVAS)
    assert (prefill["causal"], prefill["window_size"]) == (True, (W - 1, None))
    assert (commit["causal"], commit["window_size"]) == (True, (W - 1, None))
    assert (denoise["causal"], denoise["window_size"]) == (False, (W - 1, W - 1))
    assert denoise["anchor"] == ("first_query" if mode == "per_row" else "query")
    assert not prefill["anchor_passed"] and denoise["anchor_passed"] == (mode == "per_row")
    assert denoise["kv_lens"] == list(totals)
    assert denoise["max_seqlen_k"] == 2 * PAGE  # capacity, stable under CUDA graphs
    assert torch.equal(denoise["page_table"], cache.page_table[list(seq_ids)])


def test_full_layers_keep_flashinfer(monkeypatch):
    monkeypatch.setattr(gemma_attention, "require_diffusion_gemma_flashinfer",
                        lambda: (None, None, None, torch_append))
    planned = []

    class FakeWrapper:
        def plan(self, *args, **kwargs):
            planned.append(kwargs)

        def run(self, q, paged_kv_cache):
            return torch.zeros_like(q)

    monkeypatch.setattr(DiffusionGemmaPagedAttention, "_state",
                        lambda self, device: SimpleNamespace(wrapper=FakeWrapper(), plan_key=None))
    cache = DiffusionGemmaPagedKVCache(
        layer_geometries=[DiffusionGemmaLayerGeometry(HKV, D)], max_length=PAGE,
        page_size=PAGE, dtype=torch.float32, device="cpu", sliding_backend="fa4")
    kernel = ReferenceFA4()
    full = DiffusionGemmaPagedAttention(layer_id=0, num_heads=H, num_kv_heads=HKV, head_dim=D,
                                        scale=SCALE, sliding_window=None, fa4_kernel=kernel)
    metadata = cache.build_metadata(phase="prefill", seq_ids=(0,), q_offsets=(0,), q_lens=(4,),
                                    kv_lens=(4,), max_q_len=4)
    q = torch.randn(1, H, 4, D)
    full.forward(q, torch.randn(1, HKV, 4, D), torch.randn(1, HKV, 4, D), cache=cache,
                 metadata=metadata)
    assert kernel.calls == [] and len(planned) == 1


def test_fa4_cache_requires_128_token_pages():
    with pytest.raises(ValueError, match="page_size=128"):
        DiffusionGemmaPagedKVCache(layer_geometries=[DiffusionGemmaLayerGeometry(HKV, D)],
                                   max_length=PAGE, page_size=64, dtype=torch.float32,
                                   device="cpu", sliding_backend="fa4")
    flashinfer = DiffusionGemmaPagedKVCache(
        layer_geometries=[DiffusionGemmaLayerGeometry(HKV, D)], max_length=PAGE, page_size=64,
        dtype=torch.float32, device="cpu")
    metadata = flashinfer.build_metadata(phase="prefill", seq_ids=(0,), q_offsets=(0,),
                                         q_lens=(4,), kv_lens=(4,), max_q_len=4)
    assert metadata.fa4 is None


def test_graph_metadata_updates_fa4_tensors_in_place():
    cache = DiffusionGemmaPagedKVCache(
        layer_geometries=[DiffusionGemmaLayerGeometry(HKV, D)], max_length=2 * PAGE,
        page_size=PAGE, dtype=torch.float32, device="cpu", batch_size=4, sliding_backend="fa4")

    def build(phase, seq_ids, prefixes):
        return cache.build_metadata(phase=phase, seq_ids=seq_ids, q_offsets=prefixes,
                                    q_lens=(CANVAS,) * 2,
                                    kv_lens=tuple(p + CANVAS for p in prefixes), max_q_len=CANVAS)

    captured = build("denoise", (0, 1), (10, 30))
    graph = FlashInferCudaGraphRunner._make_gemma_graph_metadata(captured, cache)
    assert graph.fa4.kv_lens is not captured.fa4.kv_lens
    pointers = (graph.fa4.kv_lens.data_ptr(), graph.fa4.page_table.data_ptr())
    replay = build("denoise", (2, 3), (200, 0))
    FlashInferCudaGraphRunner._copy_gemma_metadata(graph, replay)
    assert (graph.fa4.kv_lens.data_ptr(), graph.fa4.page_table.data_ptr()) == pointers
    assert graph.fa4.kv_lens.tolist() == [208, 8]
    assert torch.equal(graph.fa4.page_table, cache.page_table[[2, 3]])
    with pytest.raises(RuntimeError, match="does not match the captured graph"):
        FlashInferCudaGraphRunner._copy_gemma_metadata(graph, build("commit", (2, 3), (200, 0)))


def _fa4_runner(monkeypatch, page_size=128, capability=(10, 0), full=None, full_backend="auto",
                device="cuda:0"):
    def init_block(self, model_config, server_args, runner_config, device, **kwargs):
        self.runner_config = runner_config
        self.device = device
        sliding = SimpleNamespace(num_heads=16, num_kv_heads=8, head_dim=256,
                                  sliding_window=1024, layer_id=0)
        full_layer = full or SimpleNamespace(num_heads=16, num_kv_heads=2, head_dim=512,
                                             sliding_window=None, layer_id=5)
        self.model = SimpleNamespace(model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=sliding), SimpleNamespace(self_attn=full_layer)]))

    probes = SimpleNamespace(flashinfer=[], fa4=[], mla=[])
    monkeypatch.setattr("fluxserve.backend.execution.runners.base.BlockDiffusionRunner.__init__",
                        init_block)
    flashinfer_runner = "fluxserve.backend.execution.runners.diffusion_gemma.flashinfer"
    fa4_runner = "fluxserve.backend.execution.runners.diffusion_gemma.fa4"
    monkeypatch.setattr(f"{flashinfer_runner}.require_diffusion_gemma_flashinfer", lambda: None)
    monkeypatch.setattr(f"{flashinfer_runner}.probe_diffusion_gemma_flashinfer",
                        lambda geometries, device: probes.flashinfer.append(geometries))
    monkeypatch.setattr(f"{fa4_runner}.validate_fa4_runtime", lambda device: None)
    monkeypatch.setattr(f"{fa4_runner}.probe_diffusion_gemma_fa4",
                        lambda geometries, device, **kw: probes.fa4.append((geometries, kw)))
    monkeypatch.setattr(f"{fa4_runner}.probe_diffusion_gemma_fa4_mla",
                        lambda geometries, device: probes.mla.append(geometries))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    config = SimpleNamespace(attention_backend="fa4", kv_cache_layout="paged", page_size=page_size,
                             enable_prefill_cuda_graph=False, enable_decode_cuda_graph=False,
                             sliding_window_mode="per_row", sliding_window_left=None,
                             full_attention_backend=full_backend)
    args = SimpleNamespace(tp_size=1, ep_size=1, dp_size=1, pp_size=1, enable_dp_attention=False)
    return DiffusionGemmaFA4Runner(SimpleNamespace(), args, config, device=device), probes


def test_fa4_runner_splits_layers_between_backends(monkeypatch):
    runner, probes = _fa4_runner(monkeypatch)
    assert runner.use_flashinfer_paged and runner.sliding_backend == "fa4"
    assert runner._gemma_page_size() == 128
    assert probes.flashinfer == [[(16, 2, 512)]]
    assert probes.fa4 == [([(16, 8, 256, 1024)],
                           dict(sliding_window_mode="per_row", sliding_window_left=None))]


def test_fa4_runner_validation(monkeypatch):
    with pytest.raises(ValueError, match="page_size=128"):
        _fa4_runner(monkeypatch, page_size=64)
    with pytest.raises(RuntimeError, match="SM100/SM110"):
        _fa4_runner(monkeypatch, capability=(12, 0))


def test_fa4_mla_full_layer_matches_dense_attention(monkeypatch):
    """Full layer as FA4 MLA: K = c * k_rot, where k_rot is V with only the rope dims changed (what
    RoPE(V) is), against dense attention written the original way, S = q . K with scale 1."""
    monkeypatch.setattr(gemma_attention, "require_diffusion_gemma_flashinfer",
                        lambda: (None, None, None, torch_append))
    c, rope_dims = 0.0623, torch.tensor([0, 1, 8, 9])  # 16-wide head: pairs (j, j + 8), j < 2
    prompts, seq_ids = (3, 20, 140), (2, 0, 1)
    totals = tuple(p + CANVAS for p in prompts)
    gen = torch.Generator().manual_seed(0)
    Q = [torch.randn(t, H, D, generator=gen) for t in totals]
    V = [torch.randn(t, HKV, D, generator=gen) for t in totals]
    K_rot = [v.clone() for v in V]
    for k in K_rot:
        k[..., rope_dims] = torch.randn(k.shape[0], HKV, len(rope_dims), generator=gen)
    cache = DiffusionGemmaPagedKVCache(
        layer_geometries=[DiffusionGemmaLayerGeometry(HKV, D, key_head_dim=len(rope_dims))],
        max_length=2 * PAGE, page_size=PAGE, dtype=torch.float32, device="cpu", batch_size=3,
        sliding_backend="fa4")
    assert cache.layer_paged_kv(0)[0].shape[-1] == len(rope_dims)
    assert cache.layer_paged_kv(0)[1].shape[-1] == D
    kernel = ReferenceFA4()
    attention = DiffusionGemmaPagedAttention(layer_id=0, num_heads=H, num_kv_heads=HKV,
                                             head_dim=D, scale=1.0, sliding_window=None,
                                             fa4_kernel=kernel)
    attention.enable_fa4_mla(scale=c, rope_dims=rope_dims)

    def run(phase, q_offsets, q_lens, max_q_len):
        metadata = cache.build_metadata(
            phase=phase, seq_ids=seq_ids, q_offsets=q_offsets, q_lens=q_lens,
            kv_lens=tuple(o + n for o, n in zip(q_offsets, q_lens)), max_q_len=max_q_len)
        parts = [slice(o, o + n) for o, n in zip(q_offsets, q_lens)]
        out = attention.forward(
            _padded([x[s] for x, s in zip(Q, parts)], H, max_q_len),
            _padded([x[s] for x, s in zip(K_rot, parts)], HKV, max_q_len),
            _padded([x[s] for x, s in zip(V, parts)], HKV, max_q_len),
            cache=cache, metadata=metadata)
        for row, (offset, n) in enumerate(zip(q_offsets, q_lens)):
            q, k, v = Q[row][offset:offset + n], c * K_rot[row][:offset + n], V[row][:offset + n]
            s = torch.einsum("qhd,khd->hqk", q, k.repeat_interleave(H // HKV, 1))
            if phase != "denoise":
                qpos = torch.arange(offset, offset + n)[:, None]
                s = s.masked_fill(torch.arange(offset + n)[None] > qpos, float("-inf"))
            expected = torch.einsum("hqk,khd->qhd", s.softmax(-1), v.repeat_interleave(H // HKV, 1))
            torch.testing.assert_close(out[row, :, :n].transpose(0, 1), expected, atol=1e-5, rtol=1e-5)
        return kernel.calls[-1]

    prefill = run("prefill", (0,) * 3, prompts, max(prompts))
    denoise = run("denoise", prompts, (CANVAS,) * 3, CANVAS)
    commit = run("commit", prompts, (CANVAS,) * 3, CANVAS)
    assert prefill["mla"] and prefill["causal"] and not denoise["causal"] and commit["causal"]
    assert denoise["softmax_scale"] == pytest.approx(c)


def test_fa4_metadata_slot_mapping_follows_static_page_table():
    cache = DiffusionGemmaPagedKVCache(
        layer_geometries=[DiffusionGemmaLayerGeometry(HKV, D)], max_length=2 * PAGE,
        page_size=PAGE, dtype=torch.float32, device="cpu", batch_size=3, sliding_backend="fa4")
    metadata = cache.build_metadata(phase="denoise", seq_ids=(2, 0), q_offsets=(130, 5),
                                    q_lens=(2, 3), kv_lens=(132, 8), max_q_len=3)
    pps = cache.pages_per_sequence
    expected = [2 * pps * PAGE + 130, 2 * pps * PAGE + 131, 5, 6, 7]
    assert metadata.fa4.slot_mapping.tolist() == expected
    flat_pages = cache.page_table.reshape(-1)
    assert [int(flat_pages[s // PAGE]) * PAGE + s % PAGE for s in expected] == expected


def _mla_attention(weight_value, rope_angles=64, head_dim=512):
    paged = SimpleNamespace(scale=1.0, mla_scale=None, mla_rope_dims=None)

    def enable(scale, rope_dims):
        paged.mla_scale, paged.mla_rope_dims = scale, rope_dims

    paged.enable_fa4_mla = enable
    return SimpleNamespace(
        num_heads=16, num_kv_heads=2, head_dim=head_dim, sliding_window=None, layer_id=5,
        k_norm=SimpleNamespace(weight=weight_value),
        rotary_emb=SimpleNamespace(rope_angles=rope_angles), paged_attention=paged)


def test_full_layer_mla_plan():
    from fluxserve.backend.execution.runners.diffusion_gemma.fa4 import full_layer_mla_plan

    scale, rope_dims = full_layer_mla_plan(_mla_attention(torch.full((512,), 0.0623)))
    assert scale == pytest.approx(0.0623)
    assert rope_dims.tolist() == list(range(64)) + list(range(256, 320))
    per_dim = torch.full((512,), 0.0623)
    per_dim[3] = 0.5
    assert "single scalar" in full_layer_mla_plan(_mla_attention(per_dim))
    assert "rotary" in full_layer_mla_plan(_mla_attention(torch.ones(512), rope_angles=None))


def test_fa4_runner_runs_full_layers_as_mla(monkeypatch):
    full = _mla_attention(torch.full((512,), 0.0623))
    runner, probes = _fa4_runner(monkeypatch, full=full, device="cpu")
    assert runner.full_layer_mla and full.paged_attention.mla_scale == pytest.approx(0.0623)
    assert probes.flashinfer == []  # no layer left on FlashInfer
    assert probes.mla == [[(16, 2, 512, tuple(range(64)) + tuple(range(256, 320)))]]
    assert runner._cache_geometry(full).key_head_dim == 128
    flashinfer_runner, _ = _fa4_runner(monkeypatch, full=_mla_attention(torch.full((512,), 0.0623)),
                                       full_backend="flashinfer", device="cpu")
    assert not flashinfer_runner.full_layer_mla
    per_dim = torch.full((512,), 0.0623)
    per_dim[0] = 1.0
    with pytest.raises(ValueError, match="single scalar"):
        _fa4_runner(monkeypatch, full=_mla_attention(per_dim), full_backend="fa4", device="cpu")
