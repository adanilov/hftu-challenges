// Challenge 01: Order Book — array + 3-level bitmap implementation.

#include "solution.h"

namespace hftu {

// ---------------------------------------------------------------------------
// Side: price-indexed quantities + hierarchical occupancy bitmap.
// ---------------------------------------------------------------------------

OrderBook::Side::Side()
    : qty(kNumPrices, 0), l0(kL0, 0), l1(kL1, 0), l2(kL2, 0) {}

void OrderBook::Side::set(int32_t price) {
    const size_t p  = static_cast<size_t>(price);
    const size_t w0 = p >> 6;
    if (l0[w0] == 0) {
        const size_t w1 = w0 >> 6;
        if (l1[w1] == 0) {
            l2[w1 >> 6] |= (1ULL << (w1 & 63));
        }
        l1[w1] |= (1ULL << (w0 & 63));
    }
    l0[w0] |= (1ULL << (p & 63));
}

void OrderBook::Side::clear(int32_t price) {
    const size_t p  = static_cast<size_t>(price);
    const size_t w0 = p >> 6;
    l0[w0] &= ~(1ULL << (p & 63));
    if (l0[w0] == 0) {
        const size_t w1 = w0 >> 6;
        l1[w1] &= ~(1ULL << (w0 & 63));
        if (l1[w1] == 0) {
            l2[w1 >> 6] &= ~(1ULL << (w1 & 63));
        }
    }
}

int32_t OrderBook::Side::lowest() const {
    for (size_t w2 = 0; w2 < kL2; ++w2) {
        const uint64_t m2 = l2[w2];
        if (!m2) continue;
        const size_t   i1 = (w2 << 6) + __builtin_ctzll(m2);
        const size_t   i0 = (i1 << 6) + __builtin_ctzll(l1[i1]);
        return static_cast<int32_t>((i0 << 6) + __builtin_ctzll(l0[i0]));
    }
    return 0;
}

int32_t OrderBook::Side::highest() const {
    for (size_t w2 = kL2; w2-- > 0;) {
        const uint64_t m2 = l2[w2];
        if (!m2) continue;
        const size_t   i1 = (w2 << 6) + 63 - __builtin_clzll(m2);
        const size_t   i0 = (i1 << 6) + 63 - __builtin_clzll(l1[i1]);
        return static_cast<int32_t>((i0 << 6) + 63 - __builtin_clzll(l0[i0]));
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Open-addressing id map (linear probing, backward-shift deletion).
// ---------------------------------------------------------------------------

namespace {
inline uint64_t mix(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}
// Is k in the ring interval (a, b]?
inline bool in_range(size_t a, size_t k, size_t b) {
    return a <= b ? (a < k && k <= b) : (a < k || k <= b);
}
} // namespace

size_t OrderBook::slot_for(uint64_t id) const {
    size_t i = mix(id) & mask_;
    while (tab_[i].used && tab_[i].id != id) i = (i + 1) & mask_;
    return i;
}

void OrderBook::grow() {
    std::vector<Entry> old = std::move(tab_);
    const size_t new_cap = (old.empty() ? 1u << 17 : old.size() << 1);
    tab_.assign(new_cap, Entry{});
    mask_  = new_cap - 1;
    count_ = 0;
    for (const Entry& e : old) {
        if (!e.used) continue;
        tab_[slot_for(e.id)] = e;
        ++count_;
    }
}

void OrderBook::erase_slot(size_t pos) {
    size_t i = pos;
    for (;;) {
        tab_[i].used = false;
        size_t j = i;
        for (;;) {
            j = (j + 1) & mask_;
            if (!tab_[j].used) return;              // chain ended
            const size_t home = mix(tab_[j].id) & mask_;
            if (!in_range(i, home, j)) break;       // can shift this one back to i
        }
        tab_[i] = tab_[j];
        i = j;
    }
}

// ---------------------------------------------------------------------------
// OrderBook
// ---------------------------------------------------------------------------

OrderBook::OrderBook() {
    grow(); // allocate initial table (1 << 17 slots)
}

void OrderBook::add_order(uint64_t id, int side, int64_t price, int64_t quantity) {
    if ((count_ + 1) * 2 > tab_.size()) grow();

    const size_t s = slot_for(id);
    if (!tab_[s].used) ++count_;
    tab_[s] = {id, static_cast<int32_t>(price), static_cast<int32_t>(quantity),
               static_cast<uint8_t>(side), true};

    Side& book = (side == 0) ? bids_ : asks_;
    int64_t& lvl = book.qty[price];
    if (lvl == 0) book.set(static_cast<int32_t>(price));
    lvl += quantity;
}

void OrderBook::cancel_order(uint64_t id) {
    const size_t s = slot_for(id);
    if (!tab_[s].used) return;

    const Entry e = tab_[s];
    Side& book = (e.side == 0) ? bids_ : asks_;
    int64_t& lvl = book.qty[e.price];
    lvl -= e.quantity;
    if (lvl <= 0) {
        lvl = 0;
        book.clear(e.price);
    }

    erase_slot(s);
    --count_;
}

int64_t OrderBook::best_bid() const {
    return bids_.highest();
}

int64_t OrderBook::best_ask() const {
    return asks_.lowest();
}

} // namespace hftu
