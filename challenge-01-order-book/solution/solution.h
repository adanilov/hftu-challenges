#pragma once
// Challenge 01: Order Book
//
// Design: prices are bounded to [1, 1'000'000], so both sides of the book are
// stored as flat price-indexed quantity arrays (O(1) add/cancel, no allocation,
// one cache line touched). best_bid / best_ask need the highest/lowest occupied
// price; a 3-level bitmap over the price range answers that in O(1) via
// find-first/last-set-bit. Order lookup by id uses an open-addressing hash map
// with linear probing + backward-shift deletion (self-contained, faster than
// std::unordered_map, robust to arbitrary uint64_t ids).

#include <cstdint>
#include <vector>

namespace hftu {

class OrderBook {
public:
    OrderBook();

    // Add a new order. side: 0=buy(bid), 1=sell(ask)
    void add_order(uint64_t id, int side, int64_t price, int64_t quantity);

    // Cancel an order by ID. No-op if ID doesn't exist.
    void cancel_order(uint64_t id);

    // Return highest bid price, or 0 if no bids.
    int64_t best_bid() const;

    // Return lowest ask price, or 0 if no asks.
    int64_t best_ask() const;

private:
    static constexpr int32_t kMaxPrice  = 1'000'000;
    static constexpr size_t  kNumPrices = static_cast<size_t>(kMaxPrice) + 1; // index by price
    static constexpr size_t  kL0 = (kNumPrices + 63) / 64; // 15626 words (1 bit / price)
    static constexpr size_t  kL1 = (kL0 + 63) / 64;        // 245 words
    static constexpr size_t  kL2 = (kL1 + 63) / 64;        // 4 words

    // One side of the book: aggregate quantity per price + occupancy bitmap.
    struct Side {
        std::vector<int64_t>  qty; // qty[price] = total resting quantity
        std::vector<uint64_t> l0, l1, l2;

        Side();
        void set(int32_t price);     // mark price occupied (0 -> nonzero)
        void clear(int32_t price);   // mark price empty   (nonzero -> 0)
        int32_t lowest() const;      // lowest occupied price, 0 if empty
        int32_t highest() const;     // highest occupied price, 0 if empty
    };

    // Open-addressing id -> resting order.
    struct Entry {
        uint64_t id;
        int32_t  price;
        int32_t  quantity;
        uint8_t  side;
        bool     used;
    };

    size_t              slot_for(uint64_t id) const; // probe to id or first empty slot
    void                erase_slot(size_t pos);      // backward-shift deletion
    void                grow();

    std::vector<Entry> tab_;
    size_t             mask_  = 0;
    size_t             count_ = 0;

    Side bids_;
    Side asks_;
};

} // namespace hftu
