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
    // Flat open-addressed (linear-probe) symbol table. Contiguous storage and
    // mask-indexing beat a node-based unordered_map: no per-lookup pointer
    // chase / cache miss, and short symbols live inline via std::string SSO.
    struct Slot {
        std::string key;
        uint32_t    val = 0;
        bool        used = false;
    };
    std::vector<Slot> slots_;
    size_t            mask_ = 0;

    static size_t hash_sv(std::string_view s) {
        uint64_t h = 1469598103934665603ULL;   // FNV-1a: cheap, good on short keys
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
        return static_cast<size_t>(h);
    }
    // Returns true and sets out on hit; leaves out untouched on miss.
    bool lookup(std::string_view s, uint32_t& out) const;
};

} // namespace hftu
