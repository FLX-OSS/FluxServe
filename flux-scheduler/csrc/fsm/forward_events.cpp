// Copyright (c) 2026 FLUX-OSS
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/token_container.h"
#include "fsm/cache_states.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "resource/allocator/kv_allocator.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/node_range.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"
#include "scheduler/operations/cache.h"

namespace {

std::vector<flux::TransferPair> BuildWriteBackPairs(const std::vector<flux::TreeNode*>& write_diff) {
    std::vector<flux::TransferPair> pages_to_transfer;
    for (flux::TreeNode* n : write_diff) {
        const auto& dev_pages = n->Device().Pages();
        const auto& host_pages = n->Host().Pages();
        for (std::size_t i = 0; i < dev_pages.size(); ++i) {
            pages_to_transfer.push_back(flux::TransferPair{dev_pages[i], host_pages[i]});
        }
    }
    return pages_to_transfer;
}

void DemoteWrittenBackDevice(flux::KVPrefixCache* kv_prefix_cache, flux::HybridPrefixCache* hybrid_prefix_cache,
                             flux::TreeNode* device_node) {
    if (kv_prefix_cache == nullptr || device_node == nullptr) return;
    kv_prefix_cache->ReleaseDeviceResourcesPresentOnHost(device_node, [hybrid_prefix_cache](flux::TreeNode* n) {
        if (hybrid_prefix_cache != nullptr) {
            hybrid_prefix_cache->OnKVDeviceDemote(n);
        }
    });
}

}  // namespace

namespace flux::fsm {

void InsertHybridCache(HybridPrefixCache* hybrid_cache,
                       const std::vector<std::span<const std::int32_t>>& full_paged_tokens,
                       std::unique_ptr<DeviceNodeRef>& device_node_ref, LocalKVAllocator* local_kv_allocator,
                       const std::vector<std::int32_t>* prefix_pages_override) {
    if (hybrid_cache == nullptr) return;

    std::vector<std::int32_t> computed_prefix_pages;
    const std::vector<std::int32_t>* prefix_pages = prefix_pages_override;
    if (prefix_pages == nullptr) {
        computed_prefix_pages = DevicePagesFromRoot(device_node_ref->Node());
        prefix_pages = &computed_prefix_pages;
    }
    std::int32_t new_page_count =
        static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages->size());
    if (new_page_count <= 0) {
        return;
    }

    OwnedPages pages_to_insert = local_kv_allocator->TakeFirst(new_page_count);
    auto insert_result = hybrid_cache->GetKVPrefixCache().Insert<ResourceType::Device>(full_paged_tokens, *prefix_pages,
                                                                                       std::move(pages_to_insert));
    device_node_ref = std::make_unique<DeviceNodeRef>(insert_result.last_node);
}

std::variant<PrefillDone, Prefilling> SchedulePrefillFirstChunkEvent::operator()(Submitted&& state) {
    std::unique_ptr<HostNodeRef> host_node_ref{nullptr};
    std::unique_ptr<DeviceNodeRef> device_node_ref{nullptr};
    if (!disable_l2_cache_ && (match_result_.host.DepthInPage() > match_result_.device.DepthInPage())) {
        host_node_ref = std::make_unique<HostNodeRef>(match_result_.host.last_node);
        kv_prefix_cache_->AllocateResourceOfType<ResourceType::Device>(
            match_result_.NodesWithout<ResourceType::Device>());
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.host.last_node);
    } else {
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.device.last_node);
    }

    auto local_kv_allocator = std::make_unique<LocalKVAllocator>(device_allocator_, tokens_this_round_);
    local_kv_allocator->Acquire(decode_input_tokens_);
    auto req_pool_index = std::make_unique<ReqPoolIndex>(req_pool_allocator_->Allocate());

    TokenContainer* token_container = state.GetTokenContainer();
    std::int32_t max_matched_pages =
        disable_l2_cache_ ? match_result_.device.DepthInPage()
                          : std::max(match_result_.device.DepthInPage(), match_result_.host.DepthInPage());
    std::int32_t window_begin = max_matched_pages * state.GetPageSize();
    TokenContainer::Window window{.begin = window_begin, .size = tokens_this_round_};

    bool is_last_chunk = (window.begin + window.size) == token_container->PrefillSize();
    if (is_last_chunk) {
        return PrefillDone{token_container,
                           state.GetPageSize(),
                           std::move(host_node_ref),
                           std::move(device_node_ref),
                           std::move(local_kv_allocator),
                           std::move(req_pool_index),
                           window,
                           decode_input_tokens_};
    }
    return Prefilling{token_container,
                      state.GetPageSize(),
                      std::move(host_node_ref),
                      std::move(device_node_ref),
                      std::move(local_kv_allocator),
                      std::move(req_pool_index),
                      window};
}

std::variant<PrefillDone, Prefilling> SchedulePrefillEvent::operator()(Prefilling&& state) {
    auto local_kv_allocator = std::move(state).TakeLocalKVAllocator();
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();

    auto paged_tokens = state.GetFullPagedTokens(false);
    std::int32_t end_of_window_pages = (state.window.begin + state.window.size) / state.GetPageSize();
    if (end_of_window_pages < static_cast<std::int32_t>(paged_tokens.size())) {
        paged_tokens.resize(end_of_window_pages);
    }
    InsertHybridCache(hybrid_prefix_cache_, paged_tokens, device_node_ref, local_kv_allocator.get());
    local_kv_allocator->Acquire(tokens_this_round_);

    TokenContainer::Window window{.begin = state.window.begin + state.window.size, .size = tokens_this_round_};
    bool is_last_chunk = (window.begin + window.size) == state.GetTokenContainer()->PrefillSize();
    if (is_last_chunk) {
        return PrefillDone{state.GetTokenContainer(),
                           state.GetPageSize(),
                           std::move(host_node_ref),
                           std::move(device_node_ref),
                           std::move(local_kv_allocator),
                           std::move(state).TakeReqPoolIndex(),
                           window,
                           reserve_num_tokens_in_next_schedule_event_};
    }
    return Prefilling{state.GetTokenContainer(),
                      state.GetPageSize(),
                      std::move(host_node_ref),
                      std::move(device_node_ref),
                      std::move(local_kv_allocator),
                      std::move(state).TakeReqPoolIndex(),
                      window};
}

Decoding ScheduleDecodeEvent::operator()(PrefillDone&& state) {
    auto local_kv_allocator = std::move(state).TakeLocalKVAllocator();
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();

    auto paged_tokens = state.GetFullPagedTokens(false);
    std::int32_t end_of_window_pages = (state.window.begin + state.window.size) / state.GetPageSize();
    if (end_of_window_pages < static_cast<std::int32_t>(paged_tokens.size())) {
        paged_tokens.resize(end_of_window_pages);
    }
    InsertHybridCache(hybrid_prefix_cache_, paged_tokens, device_node_ref, local_kv_allocator.get());

    std::int32_t reserve = state.GetReserveNumTokensInNextScheduleEvent();
    local_kv_allocator->Acquire(reserve);

    return Decoding{state.GetTokenContainer(),     state.GetPageSize(),
                    std::move(host_node_ref),      std::move(device_node_ref),
                    std::move(local_kv_allocator), std::move(state).TakeReqPoolIndex(),
                    decode_input_tokens_};
}

Decoding ScheduleDecodeEvent::operator()(Decoding&& state) {
    auto local_kv_allocator = std::move(state).TakeLocalKVAllocator();
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();

    std::int32_t reserve = state.GetReserveNumTokensInNextScheduleEvent();
    local_kv_allocator->Acquire(reserve);

    return Decoding{state.GetTokenContainer(),     state.GetPageSize(),
                    std::move(host_node_ref),      std::move(device_node_ref),
                    std::move(local_kv_allocator), std::move(state).TakeReqPoolIndex(),
                    decode_input_tokens_};
}

Decoding ScheduleDecodeFromRetractedEvent::operator()(Retracted&& state) {
    std::unique_ptr<HostNodeRef> host_node_ref{nullptr};
    std::unique_ptr<DeviceNodeRef> device_node_ref{nullptr};
    if (match_result_.host.DepthInPage() > match_result_.device.DepthInPage()) {
        host_node_ref = std::make_unique<HostNodeRef>(match_result_.host.last_node);
        if (!kv_prefix_cache_->AllocateResourceOfType<ResourceType::Device>(
                match_result_.NodesWithout<ResourceType::Device>())) {
            throw std::logic_error(
                "ScheduleDecodeFromRetractedEvent: failed to allocate device pages for host cache recovery");
        }
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.host.last_node);
    } else {
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.device.last_node);
    }
    TokenContainer* token_container = state.GetTokenContainer();
    std::int32_t page_size = state.GetPageSize();
    auto local_kv_allocator = std::move(state).TakeKVAllocator();
    auto req_pool_index = std::make_unique<ReqPoolIndex>(req_pool_allocator_->Allocate());
    local_kv_allocator->Acquire(decode_input_tokens_);
    return Decoding{token_container,
                    page_size,
                    std::move(host_node_ref),
                    std::move(device_node_ref),
                    std::move(local_kv_allocator),
                    std::move(req_pool_index),
                    decode_input_tokens_};
}

template <typename ForwardStateT>
std::variant<Draining, Finished> FinishEvent::apply(ForwardStateT&& state) {
    auto full_paged_tokens = state.GetFullPagedTokens(true);
    std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(state.GetDeviceNode());
    std::int32_t alloc_count =
        static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());

    auto local_allocator = std::move(state).TakeLocalKVAllocator();
    if (alloc_count > 0) {
        OwnedPages alloc_pages = local_allocator->TakeFirst(alloc_count);
        kv_prefix_cache_->Insert<ResourceType::Device>(full_paged_tokens, prefix_pages, std::move(alloc_pages),
                                                       page_hashes_);
    }

    MatchResult match = kv_prefix_cache_->Match(full_paged_tokens);
    if (!disable_l2_cache_ && (match.device.DepthInPage() > match.host.DepthInPage())) {
        std::vector<TreeNode*> write_diff = match.NodesWithout<ResourceType::Host>();
        std::int32_t host_pages_num = 0;
        for (TreeNode* node : write_diff) {
            host_pages_num += node->Device().NumPages();
        }
        std::unique_ptr<HostNodeRef> temp_lock = std::make_unique<HostNodeRef>(match.host.last_node);
        if (!kv_prefix_cache_->EnsureCapacityByEvict<ResourceType::Host>(host_pages_num)) {
            return Finished{};
        }
        kv_prefix_cache_->AllocateResourceOfType<ResourceType::Host>(write_diff);
        std::unique_ptr<DeviceNodeRef> device_node_ref = std::make_unique<DeviceNodeRef>(match.device.last_node);
        std::unique_ptr<HostNodeRef> host_node_ref = std::make_unique<HostNodeRef>(match.device.last_node);

        auto pages_to_transfer = BuildWriteBackPairs(write_diff);
        return Draining{std::move(pages_to_transfer), std::move(device_node_ref), std::move(host_node_ref)};
    }
    return Finished{};
}

std::variant<Draining, Finished> FinishEvent::operator()(Decoding&& state) {
    return apply(std::move(state));
}

std::variant<Draining, Finished> FinishEvent::operator()(PrefillDone&& state) {
    return apply(std::move(state));
}

WritingBack FinishEvent::operator()(Retracting&& state) {
    return static_cast<WritingBack&&>(state);
}

WritingBack CommitDrainingEvent::operator()(Draining&& state) {
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();
    return WritingBack{std::move(device_node_ref), std::move(host_node_ref)};
}

Finished WriteBackDoneEvent::operator()(WritingBack&& state) {
    TreeNode* device_node = state.DeviceNode();
    state.DropDeviceNodeRef();
    DemoteWrittenBackDevice(kv_prefix_cache_, hybrid_prefix_cache_, device_node);
    return Finished{};
}

Retracted WriteBackDoneEvent::operator()(Retracting&& state) {
    TokenContainer* token_container = state.GetTokenContainer();
    std::int32_t page_size = state.GetPageSize();
    TreeNode* device_node = state.DeviceNode();
    state.DropDeviceNodeRef();
    DemoteWrittenBackDevice(kv_prefix_cache_, hybrid_prefix_cache_, device_node);
    auto host_ref = std::move(static_cast<WritingBack&&>(state)).TakeHostNodeRef();
    std::unique_ptr<LocalKVAllocator> local_device_allocator = std::move(state).TakeKVAllocator();
    return Retracted{token_container, page_size, std::move(host_ref), std::move(local_device_allocator)};
}

Finished AbortEvent::operator()(Submitted&&) {
    return Finished{};
}

Aborting AbortEvent::operator()(Prefetching&& state) {
    return Aborting{std::move(state).TakeHostPages()};
}

Finished AbortEvent::operator()(Draining&&) {
    return Finished{};
}

Finished AbortEvent::operator()(PrefetchDone&&) {
    return Finished{};
}

Aborting AbortEvent::operator()(Aborting&& state) {
    return std::move(state);
}

Finished AbortEvent::operator()(Prefilling&&) {
    return Finished{};
}

Finished AbortEvent::operator()(PrefillDone&&) {
    return Finished{};
}

Finished AbortEvent::operator()(Decoding&&) {
    return Finished{};
}

Finished AbortEvent::operator()(Retracting&&) {
    return Finished{};
}

Finished AbortEvent::operator()(Retracted&&) {
    return Finished{};
}

template <typename ForwardStateT>
Retracting ScheduleRetractEvent::applyRetract(ForwardStateT&& state) {
    std::unique_ptr<DeviceNodeRef> device_node_ref = nullptr;
    std::unique_ptr<HostNodeRef> host_node_ref = nullptr;
    std::vector<Retracting::PagePair> pages_to_transfer;

    if (match_result_.device.DepthInPage() > match_result_.host.DepthInPage()) {
        std::vector<TreeNode*> write_diff = match_result_.NodesWithout<ResourceType::Host>();
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.device.last_node);
        if (!kv_prefix_cache_->AllocateResourceOfType<ResourceType::Host>(write_diff)) {
            throw std::logic_error("ScheduleRetractEvent: failed to allocate host pages for device cache writeback");
        }
        pages_to_transfer = BuildWriteBackPairs(write_diff);
        host_node_ref = std::make_unique<HostNodeRef>(match_result_.device.last_node);
    } else {
        host_node_ref = std::make_unique<HostNodeRef>(match_result_.device.last_node);
    }

    TokenContainer* token_container = state.GetTokenContainer();
    std::int32_t page_size = state.GetPageSize();
    auto local_allocator = std::move(state).TakeLocalKVAllocator();

    return Retracting{token_container, page_size, std::move(host_node_ref), std::move(device_node_ref),
                      std::move(local_allocator), std::move(pages_to_transfer)};
}

Retracting ScheduleRetractEvent::operator()(Decoding&& state) {
    return applyRetract(std::move(state));
}

Retracting ScheduleRetractEvent::operator()(PrefillDone&& state) {
    return applyRetract(std::move(state));
}

}  // namespace flux::fsm
