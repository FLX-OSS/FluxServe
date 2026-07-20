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

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <vector>

#include "resource/radix_tree/tree_node.h"

namespace flux {

template <typename NodePtr>
class LeafToRootRange : public std::ranges::view_interface<LeafToRootRange<NodePtr>> {
public:
    class iterator {
    public:
        using value_type = NodePtr;
        using difference_type = std::ptrdiff_t;
        using pointer = NodePtr;
        using reference = NodePtr;
        using iterator_concept = std::input_iterator_tag;
        using iterator_category = std::input_iterator_tag;

        iterator() = default;
        explicit iterator(NodePtr node) : current_{node} { skip_root(); }

        NodePtr operator*() const { return current_; }

        iterator& operator++() {
            current_ = current_->Parent();
            skip_root();
            return *this;
        }

        void operator++(int) { ++*this; }

        friend bool operator==(const iterator& it, std::default_sentinel_t) { return it.current_ == nullptr; }

    private:
        void skip_root() {
            if (current_ != nullptr && current_->IsRoot()) {
                current_ = nullptr;
            }
        }

        NodePtr current_ = nullptr;
    };

    LeafToRootRange() = default;
    explicit LeafToRootRange(NodePtr leaf) : leaf_{leaf} {}

    iterator begin() const { return iterator{leaf_}; }
    std::default_sentinel_t end() const { return {}; }

private:
    NodePtr leaf_{};
};

inline auto LeafToRoot(TreeNode* node) {
    return LeafToRootRange<TreeNode*>{node};
}
inline auto LeafToRoot(const TreeNode* node) {
    return LeafToRootRange<const TreeNode*>{node};
}

template <typename NodePtr>
std::vector<NodePtr> RootToLeaf(NodePtr node) {
    std::vector<NodePtr> path;
    for (NodePtr n : LeafToRoot(node)) {
        path.push_back(n);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

template <std::ranges::input_range R>
auto Collect(R&& range) {
    using V = std::remove_cvref_t<std::ranges::range_value_t<R>>;
    std::vector<V> out;
    for (auto&& elem : range) {
        out.push_back(elem);
    }
    return out;
}

template <std::ranges::input_range R, typename Fn>
auto FlatMap(R&& range, Fn&& fn) {
    using Ret = decltype(fn(*std::ranges::begin(range)));
    using Inner = std::remove_cvref_t<Ret>;
    using T = std::ranges::range_value_t<Inner>;
    std::vector<T> out;
    for (auto&& elem : range) {
        decltype(auto) inner = fn(elem);
        out.insert(out.end(), std::ranges::begin(inner), std::ranges::end(inner));
    }
    return out;
}

inline std::vector<std::int32_t> DevicePagesFromRoot(const TreeNode* node) {
    return FlatMap(RootToLeaf(node), [](const TreeNode* n) -> const auto& { return n->Device().Pages(); });
}

}  // namespace flux
