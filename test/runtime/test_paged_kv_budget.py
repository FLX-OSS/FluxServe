# Copyright (c) 2026 FLUX-OSS
# SPDX-License-Identifier: MIT

"""KV page sizing, which decides how much cache the server allocates.

The page size is the denominator of the memory budget, so understating it does
not waste memory -- it hands out more pages than the budget can hold, and the
server dies on the allocation. A checkpoint that declares ``head_dim`` instead
of implying it through ``hidden_size // num_attention_heads`` is exactly where
that happens, and the three official Nemotron sizes bracket the case: one where
the division is too small, one where it agrees, one where it is too large.
"""

import json
import pathlib
from types import SimpleNamespace

import pytest

from fluxserve.backend.utils.runtime_utils import (
    config_head_dim,
    paged_kv_bytes_per_page,
)

CHECKPOINTS = pathlib.Path(__file__).parent / "data" / "nemotron_checkpoints"


def pinned_config(size):
    return SimpleNamespace(**json.loads((CHECKPOINTS / size / "config.json").read_text()))


@pytest.mark.parametrize("size", ["3B", "8B", "14B"])
def test_head_dim_comes_from_the_checkpoint_not_the_division(size):
    config = pinned_config(size)
    assert config_head_dim(config) == int(config.head_dim)


def test_the_division_disagrees_where_it_matters():
    """Guards the premise of this file: if every official size happened to have
    ``hidden_size // heads == head_dim`` these tests would pass vacuously."""
    divisions = {
        size: pinned_config(size).hidden_size // pinned_config(size).num_attention_heads
        for size in ("3B", "8B", "14B")
    }
    heads = {size: pinned_config(size).head_dim for size in divisions}
    assert divisions["3B"] < heads["3B"], "3B is the size that used to overallocate"
    assert divisions["14B"] > heads["14B"]
    assert divisions["8B"] == heads["8B"]


def test_a_config_without_head_dim_still_falls_back_to_the_division():
    config = SimpleNamespace(
        hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
        num_hidden_layers=2,
    )
    assert config_head_dim(config) == 128


@pytest.mark.parametrize("size", ["3B", "8B", "14B"])
def test_bytes_per_page_matches_the_real_cache_geometry(size):
    config = pinned_config(size)
    page_size = 32
    expected = (
        2  # keys and values
        * config.num_hidden_layers
        * config.num_key_value_heads
        * config.head_dim
        * page_size
        * 2  # bfloat16
    )
    assert paged_kv_bytes_per_page(config, page_size) == expected


def test_tensor_parallelism_shards_the_kv_heads():
    config = pinned_config("14B")
    whole = paged_kv_bytes_per_page(config, 32, tp_size=1)
    sharded = paged_kv_bytes_per_page(config, 32, tp_size=4)
    assert sharded * 4 == whole


def test_understating_a_page_is_what_overruns_the_budget():
    """The failure this file exists for, in the numbers the CI task used.

    A GH200 at ``--gpu-memory-utilization 0.85`` with the 3B weights resident
    leaves roughly 74 GiB of budget. Dividing that by the understated page size
    hands out enough pages to need about 99 GiB.
    """
    config = pinned_config("3B")
    budget = 74.5 * 2**30
    correct = paged_kv_bytes_per_page(config, 32)
    understated = (
        2 * config.num_hidden_layers * config.num_key_value_heads
        * (config.hidden_size // config.num_attention_heads) * 32 * 2
    )
    assert understated < correct
    pages_from_understated = int(budget // understated)
    assert pages_from_understated * correct > budget * 1.3
    assert int(budget // correct) * correct <= budget
