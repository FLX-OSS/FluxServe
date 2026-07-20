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


#include "resource/kv_prefix_cache/cache_coordinator.h"

#include <algorithm>
#include <chrono>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "resource/radix_tree/node_range.h"
#include "resource/radix_tree/tree_node.h"
#include "scheduler/outside_events/cache.h"

namespace flux {

CacheOpSpec::CacheOpSpec() = default;
CacheOpSpec::~CacheOpSpec() = default;
CacheOpSpec::CacheOpSpec(CacheOpSpec&&) noexcept = default;
CacheOpSpec& CacheOpSpec::operator=(CacheOpSpec&&) noexcept = default;

std::vector<TreeNode*> CollectNodesByOpId(TreeNode* last_node, cache_op_id op_id) {
    return Collect(LeafToRoot(last_node) | std::views::filter([op_id](TreeNode* n) {
                       auto node_op_id = n->CacheOpId();
                       return node_op_id.has_value() && *node_op_id == op_id;
                   }));
}

std::optional<CacheOpSpec> CacheCoordinator::takeOpSpec(cache_op_id op_id) {
    auto iter = pending_ops_.find(op_id);
    if (iter == pending_ops_.end()) {
        return std::nullopt;
    }
    CacheOpSpec op = std::move(iter->second);
    pending_ops_.erase(iter);
    if (op.last_node == nullptr) {
        return std::nullopt;
    }
    op.nodes = CollectNodesByOpId(op.last_node, op_id);
    return op;
}

void CacheCoordinator::HandleEvent(const cache::WriteBackDone& event) {
    auto spec = takeOpSpec(event.op_id);
    if (!spec) return;

    auto access_time = std::chrono::steady_clock::now();
    for (TreeNode* current : spec->nodes) {
        current->Touch(access_time);
    }
    if (enable_l3_storage_ && !spec->nodes.empty()) {
        EnqueueTransfer(spec->last_node);
    }
}

void CacheCoordinator::EnqueueTransfer(TreeNode* last_node) {
    if (last_node != nullptr) {
        waiting_last_nodes_.push_back(last_node);
    }
}

std::vector<TreeNode*> CacheCoordinator::DrainTransferQueue() {
    std::vector<TreeNode*> nodes = std::move(waiting_last_nodes_);
    waiting_last_nodes_.clear();
    return nodes;
}

}  // namespace flux
