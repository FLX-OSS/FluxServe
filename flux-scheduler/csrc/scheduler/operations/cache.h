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

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "resource/types.h"

namespace flux {

struct CacheOperationBase {
    cache_op_id op_id = 0;
    std::vector<std::int32_t> src_pages;
    std::vector<std::int32_t> dst_pages;
};

struct PrefetchOperation : public CacheOperationBase {
    std::string request_id;
    std::vector<std::string> rolling_page_hashes;
};

struct BackUpOperation : public CacheOperationBase {
    std::vector<std::string> rolling_page_hashes;
};

struct TransferPair {
    std::int32_t src{-1};
    std::int32_t dst{-1};

    bool operator==(const TransferPair& other) const { return src == other.src && dst == other.dst; }
};

inline std::vector<TransferPair> ToTransferPairs(const std::vector<std::tuple<std::int32_t, std::int32_t>>& pages) {
    std::vector<TransferPair> transfers;
    transfers.reserve(pages.size());
    for (const auto& page : pages) {
        transfers.push_back(TransferPair{std::get<0>(page), std::get<1>(page)});
    }
    return transfers;
}

struct WriteBackOperation {
    cache_op_id op_id{0};
    std::vector<TransferPair> transfers;  // DEVICE->HOST KV pages.
    bool is_retract{false};

    WriteBackOperation() = default;
    WriteBackOperation(cache_op_id op_id, std::vector<std::tuple<std::int32_t, std::int32_t>> pages_to_transfer,
                       bool is_retract = false)
        : op_id{op_id}, transfers{ToTransferPairs(pages_to_transfer)}, is_retract{is_retract} {}
    WriteBackOperation(cache_op_id op_id, std::vector<TransferPair> transfers, bool is_retract = false)
        : op_id{op_id}, transfers{std::move(transfers)}, is_retract{is_retract} {}
};

struct FlatWriteBackOperation {
    std::vector<cache_op_id> op_ids;
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;
    std::vector<bool> is_retract;

    explicit FlatWriteBackOperation(const std::vector<WriteBackOperation>& ops) {
        for (const auto& op : ops) {
            std::vector<std::int32_t> src_this_op;
            std::vector<std::int32_t> dst_this_op;
            src_this_op.reserve(op.transfers.size());
            dst_this_op.reserve(op.transfers.size());
            for (const auto& transfer : op.transfers) {
                src_this_op.push_back(transfer.src);
                dst_this_op.push_back(transfer.dst);
            }
            op_ids.push_back(op.op_id);
            src_pages.push_back(std::move(src_this_op));
            dst_pages.push_back(std::move(dst_this_op));
            is_retract.push_back(op.is_retract);
        }
    }
};

struct LoadBackOperation {
    cache_op_id op_id{0};
    std::vector<TransferPair> transfers;  // HOST->DEVICE KV pages.

    LoadBackOperation() = default;
    LoadBackOperation(cache_op_id op_id, std::vector<std::tuple<std::int32_t, std::int32_t>> pages_to_transfer)
        : op_id{op_id}, transfers{ToTransferPairs(pages_to_transfer)} {}
    LoadBackOperation(cache_op_id op_id, std::vector<TransferPair> transfers)
        : op_id{op_id}, transfers{std::move(transfers)} {}
};

struct FlatLoadBackOperation {
    std::vector<cache_op_id> op_ids;
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;

    explicit FlatLoadBackOperation(const std::vector<LoadBackOperation>& ops) {
        for (const auto& op : ops) {
            std::vector<std::int32_t> src_this_op;
            std::vector<std::int32_t> dst_this_op;
            src_this_op.reserve(op.transfers.size());
            dst_this_op.reserve(op.transfers.size());
            for (const auto& transfer : op.transfers) {
                src_this_op.push_back(transfer.src);
                dst_this_op.push_back(transfer.dst);
            }
            op_ids.push_back(op.op_id);
            src_pages.push_back(std::move(src_this_op));
            dst_pages.push_back(std::move(dst_this_op));
        }
    }
};

using CacheOperation = std::variant<PrefetchOperation, FlatLoadBackOperation, BackUpOperation, FlatWriteBackOperation>;

}  // namespace flux
