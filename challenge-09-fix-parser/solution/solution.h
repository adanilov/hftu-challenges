#pragma once
// Challenge 09: FIX Parser
// Edit this file and solution.cpp to implement your solution.

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace hftu {

struct TickerEntry {
    std::string_view symbol;
    uint32_t value;
};

struct ParsedOrder {
    char     msg_type;    // 'D', '8', 'F', 'G'
    int8_t   side;        // 1=Buy, 2=Sell
    uint32_t symbol_id;   // from build() mapping
    int64_t  timestamp;   // tag 52: nanoseconds from epoch
    int64_t  price;       // price * 100000000 (8 decimal places)
    int64_t  quantity;
    bool     valid;       // false if checksum failed
};

class FixParser {
public:
    FixParser();

    // Receive symbol → ID mapping. NOT timed.
    void build(std::span<const TickerEntry> entries);

    // Parse a batch of concatenated FIX messages, appending results to out.
    // out.capacity() is guaranteed to be large enough for all messages in data.
    // You should out.push_back() or out.emplace_back() for each message.
    void parse_batch(std::string_view data, std::vector<ParsedOrder>& out);

private:
    // Fast path: symbols <= 8 bytes are packed into a uint64 key. Lookup is then
    // an 8-byte load + mask, an integer hash, and integer key compares against a
    // compact 16-byte-per-slot open-addressed table — no byte-loop hashing and
    // no string compare. Symbols longer than 8 bytes (rare) use a linear fallback.
    struct Slot {
        uint64_t key = 0;
        uint32_t val = 0;
        uint32_t used = 0;
    };
    std::vector<Slot>                        fast_;
    size_t                                   fast_mask_ = 0;
    std::vector<std::pair<std::string, uint32_t>> long_;   // symbols > 8 bytes

    // Pack up to 8 bytes little-endian (byte i -> bits [8i, 8i+8)).
    static uint64_t pack8(const char* p, size_t len) {
        uint64_t k = 0;
        for (size_t i = 0; i < len; ++i)
            k |= static_cast<uint64_t>(static_cast<unsigned char>(p[i])) << (8 * i);
        return k;
    }
    static uint64_t hash64(uint64_t k) {       // murmur3 finalizer: mixes into low bits
        k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
        k ^= k >> 33;
        return k;
    }
    // Returns true and sets out on hit; leaves out untouched on miss.
    bool lookup(std::string_view s, uint32_t& out) const;
};

} // namespace hftu
