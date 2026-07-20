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
#include <span>
#include <string>
#include <vector>

#include <openssl/sha.h>

namespace flux {

inline std::string DigestToHex(const unsigned char* digest) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(SHA256_DIGEST_LENGTH * 2);
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        out.push_back(kHex[digest[i] >> 4]);
        out.push_back(kHex[digest[i] & 0xf]);
    }
    return out;
}

inline std::vector<uint8_t> HexToBytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    bytes.reserve(hex.size() / 2);
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        uint8_t hi = (hex[i] >= 'a')   ? static_cast<uint8_t>(hex[i] - 'a' + 10)
                     : (hex[i] >= 'A') ? static_cast<uint8_t>(hex[i] - 'A' + 10)
                                       : static_cast<uint8_t>(hex[i] - '0');
        uint8_t lo = (hex[i + 1] >= 'a')   ? static_cast<uint8_t>(hex[i + 1] - 'a' + 10)
                     : (hex[i + 1] >= 'A') ? static_cast<uint8_t>(hex[i + 1] - 'A' + 10)
                                           : static_cast<uint8_t>(hex[i + 1] - '0');
        bytes.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return bytes;
}

inline std::string HashPage(std::span<const std::int32_t> tokens, const std::string& prior_hash) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);

    if (!prior_hash.empty()) {
        std::vector<uint8_t> prior_bytes = HexToBytes(prior_hash);
        SHA256_Update(&ctx, prior_bytes.data(), prior_bytes.size());
    }

    for (std::int32_t t : tokens) {
        uint32_t le = static_cast<uint32_t>(t);
        uint8_t buf[4] = {
            static_cast<uint8_t>(le),
            static_cast<uint8_t>(le >> 8),
            static_cast<uint8_t>(le >> 16),
            static_cast<uint8_t>(le >> 24),
        };
        SHA256_Update(&ctx, buf, 4);
    }

    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256_Final(digest, &ctx);
    return DigestToHex(digest);
}

inline std::vector<std::string> ComputePagedHashes(const std::vector<std::span<const std::int32_t>>& token_pages,
                                                   const std::string& prior) {
    std::vector<std::string> hashes;
    hashes.reserve(token_pages.size());
    std::string current_prior = prior;
    for (const auto& page : token_pages) {
        std::string h = HashPage(page, current_prior);
        hashes.push_back(h);
        current_prior = h;
    }
    return hashes;
}

}  // namespace flux
