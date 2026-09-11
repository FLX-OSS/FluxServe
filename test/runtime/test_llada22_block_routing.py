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

"""CPU tests for LLaDA2.2 MoE block routing.

The parity test compares ``llada2_block_routing_topk`` against the reference
``LLaDA2MoeGate`` shipped in the LLaDA2.2-flash checkpoint (cached under
``tools/llada22_ref``); it is skipped when the reference files are absent.
"""

import importlib
import os
import shutil
import sys

import logging

import pytest
import torch

from fluxserve.backend.models.llada2 import llada2_block_routing_topk

# The `reference_module` fixture lives in conftest.py: it resolves the
# checkpoint's own modeling code from FLUXSERVE_LLADA22_REF_DIR or the local
# Hugging Face cache, and skips when neither is available.


def make_ref_gate(module, *, hidden, experts, top_k, block_size, capacity, seed):
    cfg_mod = importlib.import_module(
        module.__name__.rsplit(".", 1)[0] + ".configuration_llada2_moe"
    )
    config = cfg_mod.LLaDA2MoeConfig(
        hidden_size=hidden,
        num_experts=experts,
        num_experts_per_tok=top_k,
        block_size=block_size,
        expert_capacity=capacity,
        routed_scaling_factor=2.5,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    gate = module.LLaDA2MoeGate(config)
    torch.manual_seed(seed)
    with torch.no_grad():
        gate.weight.copy_(torch.randn(experts, hidden))
        gate.expert_bias.copy_(torch.randn(experts) * 0.1)
    return gate, config


class TestBlockRoutingParity:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_reference_gate(self, reference_module, seed):
        hidden, experts, top_k, block_size, capacity = 32, 64, 8, 16, 24
        gate, config = make_ref_gate(
            reference_module,
            hidden=hidden,
            experts=experts,
            top_k=top_k,
            block_size=block_size,
            capacity=capacity,
            seed=seed,
        )
        torch.manual_seed(100 + seed)
        tokens = torch.randn(4 * block_size, hidden)

        ref_idx, ref_weight, ref_logits = gate(tokens)

        weights, ids = llada2_block_routing_topk(
            ref_logits,
            gate.expert_bias,
            top_k=top_k,
            block_size=block_size,
            expert_capacity=capacity,
            renormalize=True,
        )
        assert torch.equal(ids.long(), ref_idx.long())
        # The reference gate applies routed_scaling_factor inside the gate;
        # FluxServe applies it downstream in FusedMoE.
        torch.testing.assert_close(
            weights * config.routed_scaling_factor, ref_weight, rtol=1e-5, atol=1e-6
        )


class TestBlockRoutingProperties:
    def test_block_capacity_restricts_experts(self):
        # Block 0's tokens all love expert 0; block 1's tokens all love
        # expert 1. With capacity 1, each block may only use its own expert
        # even if the other expert scores second-best per token.
        E, bs = 4, 2
        logits = torch.full((4, E), -10.0)
        logits[0:2, 0] = 5.0
        logits[0:2, 1] = 4.0
        logits[2:4, 1] = 5.0
        logits[2:4, 0] = 4.0
        bias = torch.zeros(E)
        _, ids = llada2_block_routing_topk(
            logits, bias, top_k=1, block_size=bs, expert_capacity=1, renormalize=True
        )
        assert ids[:2].flatten().tolist() == [0, 0]
        assert ids[2:].flatten().tolist() == [1, 1]

    def test_block_score_is_max_over_tokens(self):
        # One token's strong preference is enough to admit an expert for the
        # whole block (block score = max over tokens).
        E, bs = 4, 4
        logits = torch.full((bs, E), -10.0)
        logits[:, 0] = 3.0        # everyone likes expert 0
        logits[0, 1] = 6.0        # a single token loves expert 1
        bias = torch.zeros(E)
        _, ids = llada2_block_routing_topk(
            logits, bias, top_k=2, block_size=bs, expert_capacity=2, renormalize=True
        )
        # Allowed set is {0, 1}; every token's top-2 must be within it.
        assert set(ids.flatten().tolist()) <= {0, 1}

    def test_bias_affects_selection_not_weights(self):
        E, bs = 4, 2
        logits = torch.zeros(2, E)
        bias = torch.tensor([10.0, 0.0, 0.0, 0.0])
        weights, ids = llada2_block_routing_topk(
            logits, bias, top_k=1, block_size=bs, expert_capacity=2, renormalize=False
        )
        assert (ids == 0).all()  # bias drives selection
        # ...but the returned weight is the bias-free sigmoid(0) = 0.5.
        torch.testing.assert_close(weights, torch.full_like(weights, 0.5))

    def test_misaligned_token_count_raises(self):
        with pytest.raises(ValueError, match="block_size"):
            llada2_block_routing_topk(
                torch.randn(5, 8),
                torch.zeros(8),
                top_k=2,
                block_size=4,
                expert_capacity=4,
                renormalize=True,
            )


@pytest.mark.parametrize('capacity', [None, 0, 4])
def test_config_dispatch_preserves_legacy_group_routing(monkeypatch, capacity):
    from types import SimpleNamespace
    from torch import nn
    import fluxserve.backend.models.llada2 as model

    cfg = SimpleNamespace(
        num_experts_per_tok=2, norm_topk_prob=True, hidden_size=8,
        num_shared_experts=None, routed_scaling_factor=2.5,
        score_function='sigmoid', hidden_act='silu', n_group=2,
        topk_group=1, num_experts=8, moe_intermediate_size=8,
        block_size=4,
    )
    if capacity is not None:
        cfg.expert_capacity = capacity
    monkeypatch.setattr(model, 'get_tensor_model_parallel_world_size', lambda: 1)
    monkeypatch.setattr(model, 'get_moe_a2a_backend', lambda: SimpleNamespace(is_deepep=lambda: False))
    monkeypatch.setattr(model, 'FusedMoE', lambda **kw: nn.Identity())
    monkeypatch.setattr(model, 'LLaDA2Gate', lambda **kw: SimpleNamespace(expert_bias=torch.zeros(8)))
    captured = {}

    def topk(**kwargs):
        captured.update(kwargs)
        return nn.Identity()

    monkeypatch.setattr(model, 'TopK', topk)
    block = model.LLaDA2SparseMoeBlock(1, cfg)
    assert block.use_block_routing == bool(capacity)
    assert captured['use_grouped_topk'] == (not bool(capacity))
    assert (captured['custom_routing_function'] is not None) == bool(capacity)
    if capacity:
        assert captured['output_format'] == model.TopKOutputFormat.STANDARD


def _make_moe_block(monkeypatch, *, capacity, layer_id, first_k_dense_replace):
    """Construct LLaDA2SparseMoeBlock with the expert plumbing stubbed out."""
    from types import SimpleNamespace
    from torch import nn
    import fluxserve.backend.models.llada2 as model

    cfg = SimpleNamespace(
        num_experts_per_tok=2, norm_topk_prob=True, hidden_size=8,
        num_shared_experts=None, routed_scaling_factor=2.5,
        score_function='sigmoid', hidden_act='silu', n_group=2,
        topk_group=1, num_experts=8, moe_intermediate_size=8,
        block_size=4, expert_capacity=capacity,
        first_k_dense_replace=first_k_dense_replace,
    )
    monkeypatch.setattr(model, 'get_tensor_model_parallel_world_size', lambda: 1)
    monkeypatch.setattr(model, 'get_moe_a2a_backend', lambda: SimpleNamespace(is_deepep=lambda: False))
    monkeypatch.setattr(model, 'FusedMoE', lambda **kw: nn.Identity())
    monkeypatch.setattr(model, 'LLaDA2Gate', lambda **kw: SimpleNamespace(expert_bias=torch.zeros(8)))
    monkeypatch.setattr(model, 'TopK', lambda **kw: nn.Identity())
    return model.LLaDA2SparseMoeBlock(layer_id, cfg)


class TestBlockRoutingBanner:
    """The startup banner is an acceptance signal for GPU runs.

    Job 3481121 shipped with block routing active but the banner missing: the
    `fluxserve` stdlib logger had no handler, so every model-side record was
    dropped and the one log line that evidences routing never appeared.
    """

    def test_banner_logged_on_the_first_moe_layer(self, monkeypatch, caplog):
        with caplog.at_level(logging.INFO, logger='fluxserve.backend.models.llada2'):
            _make_moe_block(monkeypatch, capacity=4, layer_id=1, first_k_dense_replace=1)
        assert 'MoE block routing active: block_size=4 expert_capacity=4' in caplog.text

    def test_banner_not_repeated_on_later_layers(self, monkeypatch, caplog):
        with caplog.at_level(logging.INFO, logger='fluxserve.backend.models.llada2'):
            _make_moe_block(monkeypatch, capacity=4, layer_id=7, first_k_dense_replace=1)
        assert 'block routing active' not in caplog.text

    def test_no_banner_when_routing_is_off(self, monkeypatch, caplog):
        with caplog.at_level(logging.INFO, logger='fluxserve.backend.models.llada2'):
            block = _make_moe_block(monkeypatch, capacity=0, layer_id=1, first_k_dense_replace=1)
        assert not block.use_block_routing
        assert 'block routing active' not in caplog.text


class TestFluxserveLoggingSetup:
    def test_attaches_one_scoped_handler_and_is_idempotent(self):
        from fluxserve.cli import configure_logging

        namespace = logging.getLogger('fluxserve')
        saved = list(namespace.handlers)
        namespace.handlers = [h for h in saved if not getattr(h, '_fluxserve_handler', False)]
        try:
            configure_logging(3)
            ours = [h for h in namespace.handlers if getattr(h, '_fluxserve_handler', False)]
            assert len(ours) == 1
            assert 'rank3' in ours[0].formatter._fmt
            assert namespace.level == logging.INFO
            configure_logging(3)
            assert len([h for h in namespace.handlers if getattr(h, '_fluxserve_handler', False)]) == 1
            # Scoped: third-party namespaces do not inherit our handler.
            assert not any(
                getattr(h, '_fluxserve_handler', False)
                for h in logging.getLogger('transformers').handlers
            )
        finally:
            namespace.handlers = saved
