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

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "scheduler/outside_events/inc.h"
#include "scheduler/operations/inc.h"
#include "scheduler/execution_event.h"
#include "scheduler/kv_cache_events.h"
#include "scheduler/request.h"
#include "scheduler/scheduler.h"
#include "scheduler/types.h"

/*
Writable types:
1. SchedulerConfig
2. RequestSpec
3. ForwardEvent
4. AbortEvent
5. cache::*DoneEvent

All other types are produced by the scheduler and consumed by Python, so they do
not need writable properties.
*/

namespace nb = nanobind;

namespace {

template <typename Op, typename Cls>
void BindForwardCommonFields(Cls& cls) {
    cls.def_prop_ro(
           "request_ids", [](const Op& op) -> const std::vector<std::string>& { return op.request_ids; },
           nb::rv_policy::reference_internal)
        .def_prop_ro(
            "request_pool_indices",
            [](const Op& op) -> const std::vector<std::int32_t>& { return op.request_pool_indices; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "input_lengths", [](const Op& op) -> const std::vector<std::int32_t>& { return op.input_lengths; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "occupied_pages",
            [](const Op& op) -> const std::vector<std::vector<std::int32_t>>& { return op.occupied_pages; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "begins", [](const Op& op) -> const std::vector<std::int32_t>& { return op.begins; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "sizes", [](const Op& op) -> const std::vector<std::int32_t>& { return op.sizes; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "new_occupied_pages",
            [](const Op& op) {
                std::vector<std::vector<std::int32_t>> result;
                result.reserve(op.occupied_pages.size());
                for (std::size_t i = 0; i < op.occupied_pages.size(); ++i) {
                    const auto& pages = op.occupied_pages[i];
                    std::int32_t b = op.begins[i];
                    std::int32_t s = op.sizes[i];
                    result.emplace_back(pages.begin() + b, pages.begin() + b + s);
                }
                return result;
            },
            nb::rv_policy::copy);
}

template <typename Op, typename Cls>
void BindCacheCommonFields(Cls& cls) {
    cls.def_prop_ro(
           "op_id", [](const Op& op) -> const flux::cache_op_id& { return op.op_id; },
           nb::rv_policy::reference_internal)
        .def_prop_ro(
            "src_pages", [](const Op& op) -> const std::vector<std::int32_t>& { return op.src_pages; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "dst_pages", [](const Op& op) -> const std::vector<std::int32_t>& { return op.dst_pages; },
            nb::rv_policy::reference_internal);
}

}  // namespace

NB_MODULE(flux_scheduler_ext, m) {
    m.doc() = "Flux scheduler bindings";

    nb::class_<flux::SchedulerStats>(m, "SchedulerStats")
        .def(nb::init<>())
        .def_ro("total_batches", &flux::SchedulerStats::total_batches)
        .def_ro("mixed_batches", &flux::SchedulerStats::mixed_batches)
        .def_ro("retract_count", &flux::SchedulerStats::retract_count)
        .def_ro("abort_count", &flux::SchedulerStats::abort_count)
        .def_ro("schedule_latency_count", &flux::SchedulerStats::schedule_latency_count)
        .def_ro("schedule_latency_sum_us", &flux::SchedulerStats::schedule_latency_sum_us)
        .def_ro("schedule_latency_max_us", &flux::SchedulerStats::schedule_latency_max_us)
        .def_ro("prefix_cache_hit_tokens", &flux::SchedulerStats::prefix_cache_hit_tokens)
        .def_ro("prefix_cache_req_tokens", &flux::SchedulerStats::prefix_cache_req_tokens)
        .def_ro("pending_queue_size", &flux::SchedulerStats::pending_queue_size)
        .def_ro("plan_queue_size", &flux::SchedulerStats::plan_queue_size)
        .def_ro("event_queue_size", &flux::SchedulerStats::event_queue_size)
        .def_ro("active_requests", &flux::SchedulerStats::active_requests);

    nb::module_ kv_event = m.def_submodule("KVEvent");
    nb::class_<flux::KvBlockStoredEvent>(kv_event, "BlockStored")
        .def_prop_ro("kind", [](const flux::KvBlockStoredEvent&) { return "BlockStored"; })
        .def_ro("block_hashes", &flux::KvBlockStoredEvent::block_hashes)
        .def_ro("parent_block_hash", &flux::KvBlockStoredEvent::parent_block_hash)
        .def_ro("token_ids", &flux::KvBlockStoredEvent::token_ids)
        .def_ro("block_size", &flux::KvBlockStoredEvent::block_size);

    nb::class_<flux::KvBlockRemovedEvent>(kv_event, "BlockRemoved")
        .def_prop_ro("kind", [](const flux::KvBlockRemovedEvent&) { return "BlockRemoved"; })
        .def_ro("block_hashes", &flux::KvBlockRemovedEvent::block_hashes);

    auto scheduler_config = nb::class_<flux::SchedulerConfig>(m, "SchedulerConfig");

    nb::enum_<flux::PagedCacheGroupConfig::Retention>(m, "PagedCacheRetention")
        .value("FullHistory", flux::PagedCacheGroupConfig::Retention::FullHistory)
        .value("SlidingWindow", flux::PagedCacheGroupConfig::Retention::SlidingWindow);

    nb::enum_<flux::PagedCacheGroupFamily>(m, "PagedCacheGroupFamily")
        .value("History", flux::PagedCacheGroupFamily::History)
        .value("State", flux::PagedCacheGroupFamily::State);

    nb::class_<flux::PagedCacheGroupConfig>(m, "PagedCacheGroupConfig")
        .def(nb::init<>())
        .def(
            "__init__",
            [](flux::PagedCacheGroupConfig* self, std::string group_id, std::int32_t rows_per_page,
               std::int32_t entry_stride_tokens, std::int32_t total_pages,
               flux::PagedCacheGroupConfig::Retention retention,
               std::optional<std::int32_t> sliding_window_tokens, flux::PagedCacheGroupFamily family) {
                new (self) flux::PagedCacheGroupConfig{
                    std::move(group_id),   rows_per_page, entry_stride_tokens, total_pages, retention,
                    sliding_window_tokens, family};
            },
            nb::arg("group_id"), nb::arg("rows_per_page"), nb::arg("entry_stride_tokens"), nb::arg("total_pages"),
            nb::arg("retention") = flux::PagedCacheGroupConfig::Retention::FullHistory,
            nb::arg("sliding_window_tokens") = std::nullopt,
            nb::arg("family") = flux::PagedCacheGroupFamily::History)
        .def_rw("group_id", &flux::PagedCacheGroupConfig::group_id)
        .def_rw("rows_per_page", &flux::PagedCacheGroupConfig::rows_per_page)
        .def_rw("entry_stride_tokens", &flux::PagedCacheGroupConfig::entry_stride_tokens)
        .def_rw("total_pages", &flux::PagedCacheGroupConfig::total_pages)
        .def_rw("retention", &flux::PagedCacheGroupConfig::retention)
        .def_rw("sliding_window_tokens", &flux::PagedCacheGroupConfig::sliding_window_tokens)
        .def_rw("family", &flux::PagedCacheGroupConfig::family)
        .def("raw_tokens_per_page", &flux::PagedCacheGroupConfig::RawTokensPerPage)
        .def("validate", &flux::PagedCacheGroupConfig::Validate);

    nb::class_<flux::PagedCacheGroupAllocator>(m, "PagedCacheGroupAllocator")
        .def(nb::init<flux::PagedCacheGroupConfig>(), nb::arg("config"))
        .def("allocate", &flux::PagedCacheGroupAllocator::Allocate, nb::arg("num_pages"))
        .def("deallocate", &flux::PagedCacheGroupAllocator::Deallocate, nb::arg("pages"))
        .def("config", &flux::PagedCacheGroupAllocator::Config, nb::rv_policy::reference_internal)
        .def("total_pages", &flux::PagedCacheGroupAllocator::TotalPages)
        .def("available_pages", &flux::PagedCacheGroupAllocator::AvailablePages)
        .def("allocated_pages_total", &flux::PagedCacheGroupAllocator::AllocatedPagesTotal)
        .def("released_pages_total", &flux::PagedCacheGroupAllocator::ReleasedPagesTotal)
        .def("failed_alloc_count", &flux::PagedCacheGroupAllocator::FailedAllocCount);

    nb::class_<flux::PagedCacheGroupTable>(m, "PagedCacheGroupTable")
        .def(nb::init<flux::PagedCacheGroupAllocator*>(), nb::arg("allocator"), nb::keep_alive<1, 2>())
        .def("acquire", &flux::PagedCacheGroupTable::Acquire, nb::arg("target_raw_tokens_exclusive"))
        .def("release_skipped", &flux::PagedCacheGroupTable::ReleaseSkipped, nb::arg("window_lower_bound"))
        .def("release_all", &flux::PagedCacheGroupTable::ReleaseAll)
        .def("page_ids", &flux::PagedCacheGroupTable::PageIds, nb::rv_policy::reference_internal)
        .def("size", &flux::PagedCacheGroupTable::Size)
        .def("active_pages_count", &flux::PagedCacheGroupTable::ActivePagesCount)
        .def("owned_pages_count", &flux::PagedCacheGroupTable::OwnedPagesCount)
        .def("borrowed_pages_count", &flux::PagedCacheGroupTable::BorrowedPagesCount)
        .def("released_pages_count", &flux::PagedCacheGroupTable::ReleasedPagesCount)
        .def("base_logical_page", &flux::PagedCacheGroupTable::BaseLogicalPage)
        .def("raw_token_cursor", &flux::PagedCacheGroupTable::RawTokenCursor)
        .def("rows_per_page", &flux::PagedCacheGroupTable::RowsPerPage)
        .def("entry_stride_tokens", &flux::PagedCacheGroupTable::EntryStrideTokens)
        .def("raw_tokens_per_page", &flux::PagedCacheGroupTable::RawTokensPerPage)
        .def("is_sliding", &flux::PagedCacheGroupTable::IsSliding)
        .def("sliding_window_tokens", &flux::PagedCacheGroupTable::SlidingWindowTokens);

    // Python declares the required group ids only. Scheduler derives LCM and
    // sliding-window metadata from the matching PagedCacheGroupConfig entries.
    nb::class_<flux::PrefixCacheAdjunctSpec>(m, "PrefixCacheAdjunctSpec")
        .def(nb::init<>())
        .def_rw("required_groups", &flux::PrefixCacheAdjunctSpec::required_groups);

    scheduler_config.def(nb::init<>())
        .def_rw("page_size", &flux::SchedulerConfig::page_size)
        .def_rw("max_scheduled_tokens", &flux::SchedulerConfig::max_scheduled_tokens)
        .def_rw("max_batch_size", &flux::SchedulerConfig::max_batch_size)
        .def_rw("decode_input_tokens", &flux::SchedulerConfig::decode_input_tokens)
        .def_prop_rw(
            "num_device_pages", [](const flux::SchedulerConfig& c) { return c.device_allocator.total_pages; },
            [](flux::SchedulerConfig& c, std::int32_t v) { c.device_allocator.total_pages = v; })
        .def_prop_rw(
            "num_host_pages", [](const flux::SchedulerConfig& c) { return c.host_allocator.total_pages; },
            [](flux::SchedulerConfig& c, std::int32_t v) { c.host_allocator.total_pages = v; })
        .def_rw("paged_cache_groups", &flux::SchedulerConfig::paged_cache_groups)
        .def_rw("prefix_cache_adjunct", &flux::SchedulerConfig::prefix_cache_adjunct)
        .def_rw("disable_l2_cache", &flux::SchedulerConfig::disable_l2_cache)
        .def_rw("enable_l3_storage", &flux::SchedulerConfig::enable_l3_storage)
        .def_rw("prefetch_threshold", &flux::SchedulerConfig::prefetch_threshold)
        .def_rw("enable_kv_cache_events", &flux::SchedulerConfig::enable_kv_cache_events)
        .def_rw("enable_mixed_prefill_decode", &flux::SchedulerConfig::enable_mixed_prefill_decode)
        .def_rw("disable_prefix_cache", &flux::SchedulerConfig::disable_prefix_cache);

    nb::class_<flux::RequestSpec>(m, "RequestSpec")
        .def(nb::init<>())
        .def_rw("request_id", &flux::RequestSpec::request_id)
        .def_rw("tokens", &flux::RequestSpec::tokens)
        .def_rw("prefill_length", &flux::RequestSpec::prefill_length)
        .def_rw("rolling_hashes", &flux::RequestSpec::rolling_hashes)
        .def_rw("storage_hit_pages", &flux::RequestSpec::storage_hit_pages);

    nb::module_ forward_event = m.def_submodule("ForwardEvent");
    nb::class_<flux::forward::ExtendResult>(forward_event, "ExtendResult")
        .def(nb::init<>())
        .def_rw("request_id", &flux::forward::ExtendResult::request_id)
        .def_rw("tokens", &flux::forward::ExtendResult::tokens);

    nb::class_<flux::forward::Finish>(forward_event, "Finish")
        .def(nb::init<>())
        .def_rw("request_id", &flux::forward::Finish::request_id);

    nb::class_<flux::forward::Abort>(forward_event, "Abort")
        .def(nb::init<>())
        .def_rw("request_id", &flux::forward::Abort::request_id);

    nb::class_<flux::forward::UpdateReserveNumTokens>(forward_event, "UpdateReserveNumTokens")
        .def(nb::init<>())
        .def_rw("request_id", &flux::forward::UpdateReserveNumTokens::request_id)
        .def_rw("reserve_num_tokens_in_next_schedule_event",
                &flux::forward::UpdateReserveNumTokens::reserve_num_tokens_in_next_schedule_event);

    // ─── ExecutionEvent ─────────────────────────────────────────────

    nb::module_ cache = m.def_submodule("Cache");

    nb::class_<flux::cache::PrefetchDone>(cache, "PrefetchDoneEvent")
        .def(nb::init<>())
        .def_rw("success", &flux::cache::PrefetchDone::success)
        .def_rw("op_id", &flux::cache::PrefetchDone::op_id)
        .def_rw("request_id", &flux::cache::PrefetchDone::request_id)
        .def_rw("completed_pages", &flux::cache::PrefetchDone::completed_pages);

    nb::class_<flux::cache::WriteBackDone>(cache, "WriteBackDoneEvent")
        .def(nb::init<>())
        .def_rw("op_id", &flux::cache::WriteBackDone::op_id)
        .def_rw("success", &flux::cache::WriteBackDone::success);

    nb::class_<flux::ExecutionEvent>(m, "ExecutionEvent")
        .def(nb::init<>())
        .def(
            "add_event",
            [](flux::ExecutionEvent& self, flux::Event e) -> flux::ExecutionEvent& {
                return self.With(std::move(e));
            },
            nb::arg("event"), nb::rv_policy::reference);

    nb::module_ forward = m.def_submodule("Forward");

    auto flat_fwd_op = nb::class_<flux::FlatForwardOperation>(forward, "FlatForwardOp");
    BindForwardCommonFields<flux::FlatForwardOperation>(flat_fwd_op);
    flat_fwd_op.def_ro("input_ids", &flux::FlatForwardOperation::input_ids)
        .def_ro("shifted_input_ids", &flux::FlatForwardOperation::shifted_input_ids)
        .def_ro("extend_prefix_lens", &flux::FlatForwardOperation::extend_prefix_lens)
        .def_prop_ro(
            "prefill_lengths",
            [](const flux::FlatForwardOperation& op) -> const std::vector<std::int32_t>& {
                return op.prefill_lengths;
            },
            nb::rv_policy::reference_internal)
        .def_ro("decode_input_ids", &flux::FlatForwardOperation::decode_input_ids)
        .def_rw("hist_token_lens", &flux::FlatForwardOperation::hist_token_lens)
        .def_prop_ro(
            "paged_cache_block_tables",
            [](const flux::FlatForwardOperation& op)
                -> const std::map<std::string, std::vector<std::vector<std::int32_t>>>& {
                return op.paged_cache_block_tables;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "paged_cache_block_table_base_offsets",
            [](const flux::FlatForwardOperation& op) -> const std::map<std::string, std::vector<std::int32_t>>& {
                return op.paged_cache_block_table_base_offsets;
            },
            nb::rv_policy::reference_internal)
        .def("num_extends", &flux::FlatForwardOperation::num_extends);

    // ─── CacheOperation (attached to the Cache submodule) ──────────
    auto prefetch_op = nb::class_<flux::PrefetchOperation>(cache, "PrefetchOp");
    BindCacheCommonFields<flux::PrefetchOperation>(prefetch_op);
    prefetch_op.def(nb::init<>())
        .def_ro("request_id", &flux::PrefetchOperation::request_id)
        .def_ro("rolling_page_hashes", &flux::PrefetchOperation::rolling_page_hashes);

    auto backup_op = nb::class_<flux::BackUpOperation>(cache, "BackUpOp");
    BindCacheCommonFields<flux::BackUpOperation>(backup_op);
    backup_op.def(nb::init<>()).def_ro("rolling_page_hashes", &flux::BackUpOperation::rolling_page_hashes);

    nb::class_<flux::FlatLoadBackOperation>(cache, "LoadBackOp")
        .def_ro("op_ids", &flux::FlatLoadBackOperation::op_ids)
        .def_ro("src_pages", &flux::FlatLoadBackOperation::src_pages)
        .def_ro("dst_pages", &flux::FlatLoadBackOperation::dst_pages);

    nb::class_<flux::FlatWriteBackOperation>(cache, "WriteBackOp")
        .def_ro("op_ids", &flux::FlatWriteBackOperation::op_ids)
        .def_ro("src_pages", &flux::FlatWriteBackOperation::src_pages)
        .def_ro("dst_pages", &flux::FlatWriteBackOperation::dst_pages)
        .def_ro("is_retract", &flux::FlatWriteBackOperation::is_retract);

    auto collect_forward = [](const flux::ExecutionPlan& plan) -> nb::list {
        nb::list result;
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<flux::FlatForwardOperation>(&op)) {
                result.append(nb::cast(*f, nb::rv_policy::copy));
            }
        }
        return result;
    };

    auto collect_cache = [](const flux::ExecutionPlan& plan) -> nb::list {
        nb::list result;
        for (const auto& op : plan.Operations()) {
            if (auto* c = std::get_if<flux::CacheOperation>(&op)) {
                std::visit([&result](const auto& inner) { result.append(nb::cast(inner, nb::rv_policy::copy)); }, *c);
            }
        }
        return result;
    };

    nb::class_<flux::ExecutionPlan>(m, "ExecutionPlan")
        .def(nb::init<>())
        .def_prop_ro("forward", collect_forward)
        .def_prop_ro("cache", collect_cache);

    nb::class_<flux::Scheduler>(m, "Scheduler")
        .def(nb::init<flux::SchedulerConfig>(), nb::arg("config") = flux::SchedulerConfig{})
        .def("submit_requests",
             nb::overload_cast<const std::vector<flux::RequestSpec>&>(&flux::Scheduler::SubmitRequests),
             nb::arg("request_specs"))
        .def("next_execution_plan", [](flux::Scheduler& s) { return s.NextExecutionPlan(); })
        .def("advance", &flux::Scheduler::Advance, nb::arg("event"))
        .def(
            "drain_kv_events",
            [](flux::Scheduler& s) {
                nb::list result;
                for (auto& event : s.DrainKvEvents()) {
                    std::visit([&result](auto& inner) { result.append(nb::cast(inner, nb::rv_policy::copy)); }, event);
                }
                return result;
            },
            nb::rv_policy::move)
        .def("waiting_size", &flux::Scheduler::WaitingSize)
        .def("decoding_size", &flux::Scheduler::DecodingSize)
        .def("prefilling_size", &flux::Scheduler::PrefillSize)
        .def("retract_count", &flux::Scheduler::RetractedSize)
        .def("available_kv_pages", &flux::Scheduler::AvailableKvPages)
        .def("active_kv_pages", &flux::Scheduler::ActiveKvPages)
        .def("get_request_token_size", &flux::Scheduler::GetRequestTokenSize, nb::arg("id"))
        .def("calc_rolling_hash", &flux::Scheduler::CalcRollingHash, nb::arg("input_tokens"),
             nb::arg("apply_match") = false)
        .def("paged_cache_group_ids", &flux::Scheduler::PagedCacheGroupIds)
        .def("paged_cache_group_total_pages", &flux::Scheduler::PagedCacheGroupTotalPages, nb::arg("group_id"))
        .def("paged_cache_group_available_pages", &flux::Scheduler::PagedCacheGroupAvailablePages,
             nb::arg("group_id"))
        .def("paged_cache_group_failed_alloc_count", &flux::Scheduler::PagedCacheGroupFailedAllocCount,
             nb::arg("group_id"))
        .def("get_request_paged_cache_page_ids", &flux::Scheduler::GetRequestPagedCachePageIds,
             nb::arg("request_id"), nb::arg("group_id"))
        .def("get_request_paged_cache_base_logical_page", &flux::Scheduler::GetRequestPagedCacheBaseLogicalPage,
             nb::arg("request_id"), nb::arg("group_id"));
}
