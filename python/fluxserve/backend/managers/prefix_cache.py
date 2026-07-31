# Copyright (c) 2026 FLUX-OSS

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrefixCacheStats:
    lookups: int = 0
    full_hits: int = 0
    partial_hits: int = 0
    misses: int = 0
    hit_tokens: int = 0
    evictions: int = 0


class _Node:
    def __init__(self, parent=None, key=None, page_id=None):
        self.parent = parent
        self.key = key
        self.page_id = page_id
        self.children = {}
        self.references = 0
        self.last_access = 0


@dataclass
class PrefixCacheLease:
    page_ids: list[int]
    matched_pages: int
    cacheable_pages: int
    nodes: list[_Node]
    released: bool = False

    @property
    def matched_tokens(self) -> int:
        return self.matched_pages * self._page_size

    _page_size: int = 1


class PrefixCacheManager:
    """Page-aligned radix cache and physical-page allocator.

    Metadata stays on CPU. Page IDs address a persistent device KV pool owned by
    the caller, which makes this manager usable by both offline and online paths.
    """

    def __init__(self, *, page_size: int, num_pages: int, reserved_pages=(0,)):
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        reserved = {int(page) for page in reserved_pages}
        if num_pages <= len(reserved):
            raise ValueError("num_pages must exceed the reserved page count")
        if any(page < 0 or page >= num_pages for page in reserved):
            raise ValueError("reserved page IDs must be inside the page pool")
        self.page_size = int(page_size)
        self.num_pages = int(num_pages)
        self._reserved = reserved
        self.stats = PrefixCacheStats()
        self.reset()

    def reset(self):
        self.root = _Node()
        self._free_pages = set(range(self.num_pages)) - self._reserved
        self._clock = 0
        self.stats = PrefixCacheStats()

    def acquire(
        self, token_ids, *, cacheable_length: int, required_pages: int
    ) -> PrefixCacheLease:
        tokens = [int(token) for token in token_ids]
        cacheable_pages = min(len(tokens), int(cacheable_length)) // self.page_size
        if required_pages < cacheable_pages:
            raise ValueError("required_pages cannot be smaller than cacheable pages")

        self.stats.lookups += 1
        node = self.root
        nodes = []
        page_ids = []
        for page_idx in range(cacheable_pages):
            start = page_idx * self.page_size
            key = tuple(tokens[start : start + self.page_size])
            child = node.children.get(key)
            if child is None:
                break
            node = child
            nodes.append(node)
            page_ids.append(node.page_id)

        matched_pages = len(page_ids)
        if matched_pages == cacheable_pages and cacheable_pages:
            self.stats.full_hits += 1
        elif matched_pages:
            self.stats.partial_hits += 1
        else:
            self.stats.misses += 1
        self.stats.hit_tokens += matched_pages * self.page_size

        for matched_node in nodes:
            matched_node.references += 1
            self._touch(matched_node)
        try:
            page_ids.extend(self._allocate(required_pages - matched_pages))
        except Exception:
            for matched_node in nodes:
                matched_node.references -= 1
            raise
        return PrefixCacheLease(
            page_ids=page_ids,
            matched_pages=matched_pages,
            cacheable_pages=cacheable_pages,
            nodes=nodes,
            _page_size=self.page_size,
        )

    def commit(self, lease: PrefixCacheLease, token_ids) -> None:
        self._require_active(lease)
        tokens = [int(token) for token in token_ids]
        node = self.root
        committed_nodes = []
        for page_idx in range(lease.cacheable_pages):
            start = page_idx * self.page_size
            key = tuple(tokens[start : start + self.page_size])
            child = node.children.get(key)
            if child is None:
                child = _Node(node, key, lease.page_ids[page_idx])
                node.children[key] = child
            elif child.page_id != lease.page_ids[page_idx]:
                self._free_pages.add(lease.page_ids[page_idx])
                lease.page_ids[page_idx] = child.page_id
            node = child
            committed_nodes.append(node)
            self._touch(node)

        old_nodes = set(lease.nodes)
        for old_node in old_nodes:
            old_node.references -= 1
        for committed_node in committed_nodes:
            committed_node.references += 1
        lease.nodes = committed_nodes
        lease.matched_pages = lease.cacheable_pages

    def release(self, lease: PrefixCacheLease) -> None:
        if lease.released:
            return
        cached_pages = {node.page_id for node in lease.nodes}
        for node in lease.nodes:
            node.references -= 1
            if node.references < 0:
                raise RuntimeError("prefix-cache reference count underflow")
        for page_id in lease.page_ids:
            if page_id not in cached_pages:
                self._free_pages.add(page_id)
        lease.released = True

    def snapshot(self) -> dict[str, int]:
        resident = self.num_pages - len(self._reserved) - len(self._free_pages)
        active = sum(node.references > 0 for node in self._walk_nodes())
        return {
            **vars(self.stats),
            "resident_pages": resident,
            "active_pages": active,
            "free_pages": len(self._free_pages),
        }

    def _allocate(self, count: int) -> list[int]:
        while len(self._free_pages) < count:
            if not self._evict_one():
                raise RuntimeError(
                    f"prefix cache needs {count} pages but only "
                    f"{len(self._free_pages)} are available after eviction"
                )
        pages = sorted(self._free_pages)[:count]
        self._free_pages.difference_update(pages)
        return pages

    def _evict_one(self) -> bool:
        leaves = [
            node
            for node in self._walk_nodes()
            if not node.children and node.references == 0
        ]
        if not leaves:
            return False
        victim = min(leaves, key=lambda node: (node.last_access, node.page_id))
        del victim.parent.children[victim.key]
        self._free_pages.add(victim.page_id)
        self.stats.evictions += 1
        return True

    def _walk_nodes(self):
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def _touch(self, node):
        self._clock += 1
        node.last_access = self._clock

    @staticmethod
    def _require_active(lease):
        if lease.released:
            raise RuntimeError("prefix-cache lease has already been released")
