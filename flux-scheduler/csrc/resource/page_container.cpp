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


#include "resource/page_container.h"

#include <algorithm>

#include "resource/allocator/kv_allocator.h"
#include "resource/radix_tree/node_range.h"

namespace flux {

std::vector<std::int32_t> PageContainer::Pages() const {
    std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(node_);
    std::vector<std::int32_t> local_pages = local_kv_allocator_->Pages();

    prefix_pages.insert(prefix_pages.end(), local_pages.begin(), local_pages.end());
    return prefix_pages;
}

}  // namespace flux
