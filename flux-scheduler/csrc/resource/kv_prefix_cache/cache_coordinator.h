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


#pragma once

#include <optional>
#include <unordered_map>
#include <vector>

#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"

namespace flux {

namespace cache {
struct WriteBackDone;
}

class CacheCoordinator {
public:
    explicit CacheCoordinator(bool enable_l3_storage = false) : enable_l3_storage_(enable_l3_storage) {}

    void HandleEvent(const cache::WriteBackDone& event);

    void EnqueueTransfer(TreeNode* last_node);
    std::vector<TreeNode*> DrainTransferQueue();

private:
    std::optional<CacheOpSpec> takeOpSpec(cache_op_id op_id);

    bool enable_l3_storage_;
    std::unordered_map<cache_op_id, CacheOpSpec> pending_ops_;
    std::vector<TreeNode*> waiting_last_nodes_;
};

}  // namespace flux
