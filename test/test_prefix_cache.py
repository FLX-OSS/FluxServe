import pytest

from fluxserve.backend.managers.prefix_cache import PrefixCacheManager


def _cache(num_pages=12):
    return PrefixCacheManager(page_size=2, num_pages=num_pages)


def test_exact_and_partial_prefix_hits():
    cache = _cache()
    first = cache.acquire([1, 2, 3, 4], cacheable_length=4, required_pages=4)
    cache.commit(first, [1, 2, 3, 4])
    cached_pages = first.page_ids[:2]
    cache.release(first)

    exact = cache.acquire([1, 2, 3, 4], cacheable_length=4, required_pages=4)
    assert exact.matched_tokens == 4
    assert exact.page_ids[:2] == cached_pages
    cache.release(exact)

    partial = cache.acquire([1, 2, 9, 9], cacheable_length=4, required_pages=4)
    assert partial.matched_tokens == 2
    assert partial.page_ids[0] == cached_pages[0]
    cache.release(partial)


def test_partial_pages_are_not_cached():
    cache = _cache()
    lease = cache.acquire([1, 2, 3], cacheable_length=3, required_pages=3)
    cache.commit(lease, [1, 2, 3])
    cache.release(lease)

    hit = cache.acquire([1, 2, 3], cacheable_length=3, required_pages=3)
    assert hit.matched_tokens == 2
    cache.release(hit)


def test_duplicate_insert_uses_canonical_pages():
    cache = _cache()
    first = cache.acquire([1, 2], cacheable_length=2, required_pages=2)
    second = cache.acquire([1, 2], cacheable_length=2, required_pages=2)
    cache.commit(first, [1, 2])
    canonical = first.page_ids[0]
    cache.commit(second, [1, 2])
    assert second.page_ids[0] == canonical
    cache.release(first)
    cache.release(second)


def test_lru_eviction_never_evicts_active_prefix():
    cache = _cache(num_pages=5)
    first = cache.acquire([1, 2], cacheable_length=2, required_pages=1)
    cache.commit(first, [1, 2])
    cache.release(first)
    second = cache.acquire([3, 4], cacheable_length=2, required_pages=1)
    cache.commit(second, [3, 4])
    cache.release(second)

    protected = cache.acquire([1, 2], cacheable_length=2, required_pages=1)
    allocation = cache.acquire([5, 6], cacheable_length=2, required_pages=3)
    assert cache.snapshot()["evictions"] == 1
    cache.release(allocation)
    cache.release(protected)


def test_capacity_failure_with_only_active_pages():
    cache = _cache(num_pages=3)
    active = cache.acquire([], cacheable_length=0, required_pages=2)
    with pytest.raises(RuntimeError, match="available after eviction"):
        cache.acquire([], cacheable_length=0, required_pages=1)
    cache.release(active)
