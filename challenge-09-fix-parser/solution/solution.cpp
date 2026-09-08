// Challenge 09: FIX Parser — Skeleton Implementation
// This is a correct but slow reference. You can do MUCH better!

#include "solution.h"

namespace hftu {

FixParser::FixParser() {}

void FixParser::build(std::span<const TickerEntry> entries) {
    // Size to the next power of two >= 2*N so the table stays <=50% full
    // (short probe chains). mask_ replaces a modulo on every lookup.
    size_t cap = 16;
    while (cap < entries.size() * 2) cap <<= 1;
    slots_.assign(cap, Slot{});
    mask_ = cap - 1;

    for (const auto& e : entries) {
        size_t idx = hash_sv(e.symbol) & mask_;
        while (slots_[idx].used) idx = (idx + 1) & mask_;   // linear probe
        slots_[idx].key.assign(e.symbol);
        slots_[idx].val = e.value;
        slots_[idx].used = true;
    }
}

bool FixParser::lookup(std::string_view s, uint32_t& out) const {
    size_t idx = hash_sv(s) & mask_;
    while (slots_[idx].used) {
        if (slots_[idx].key == s) { out = slots_[idx].val; return true; }
        idx = (idx + 1) & mask_;
    }
    return false;
}

// Parse a fixed-point decimal string like "187.50" into int64_t * 10^8
static int64_t parse_price(std::string_view s) {
    int64_t whole = 0;
    int64_t frac = 0;
    int frac_digits = 0;
    bool in_frac = false;

    for (char c : s) {
        if (c == '.') {
            in_frac = true;
        } else if (c >= '0' && c <= '9') {
            if (in_frac) {
                frac = frac * 10 + (c - '0');
                ++frac_digits;
            } else {
                whole = whole * 10 + (c - '0');
            }
        }
    }
    int64_t result = whole * 100'000'000LL;
    for (int i = frac_digits; i < 8; ++i) frac *= 10;
    for (int i = 8; i < frac_digits; ++i) frac /= 10;
    return result + frac;
}

static int64_t parse_int(std::string_view s) {
    int64_t val = 0;
    for (char c : s) {
        if (c >= '0' && c <= '9') val = val * 10 + (c - '0');
    }
    return val;
}

// Parse FIX timestamp "YYYYMMDD-HH:MM:SS.nnnnnnnnn" to nanoseconds from epoch
static int64_t parse_timestamp(std::string_view s) {
    // Minimal parsing: extract components
    // Format: 20260320-14:30:00.123456789
    if (s.size() < 17) return 0;

    int year = 0, month = 0, day = 0, hour = 0, min = 0, sec = 0;
    int64_t nanos = 0;

    // YYYYMMDD
    for (int i = 0; i < 4; ++i) year = year * 10 + (s[i] - '0');
    for (int i = 4; i < 6; ++i) month = month * 10 + (s[i] - '0');
    for (int i = 6; i < 8; ++i) day = day * 10 + (s[i] - '0');
    // HH:MM:SS
    hour = (s[9] - '0') * 10 + (s[10] - '0');
    min = (s[12] - '0') * 10 + (s[13] - '0');
    sec = (s[15] - '0') * 10 + (s[16] - '0');

    // Fractional seconds (variable length)
    if (s.size() > 17 && s[17] == '.') {
        int frac_digits = 0;
        for (size_t i = 18; i < s.size() && s[i] >= '0' && s[i] <= '9'; ++i) {
            nanos = nanos * 10 + (s[i] - '0');
            ++frac_digits;
        }
        // Scale to nanoseconds (9 digits)
        for (int i = frac_digits; i < 9; ++i) nanos *= 10;
    }

    // Convert to days since epoch (simplified, no leap second handling)
    // Using a basic algorithm for days since 1970-01-01
    auto days_since_epoch = [](int y, int m, int d) -> int64_t {
        if (m <= 2) { y--; m += 12; }
        int64_t days = 365LL * y + y / 4 - y / 100 + y / 400 + (153 * (m - 3) + 2) / 5 + d - 719469;
        return days;
    };

    int64_t total_secs = days_since_epoch(year, month, day) * 86400LL +
                         hour * 3600LL + min * 60LL + sec;
    return total_secs * 1'000'000'000LL + nanos;
}

void FixParser::parse_batch(std::string_view data, std::vector<ParsedOrder>& out) {
    const char* const end = data.data() + data.size();
    const char* p = data.data();

    while (p < end) {
        ParsedOrder order{};
        unsigned running_cs = 0;   // sum of every byte seen in this message
        bool msg_complete = false;

        // One linear pass over the message. Each iteration consumes exactly
        // one "tag=value<SOH>" field and folds its bytes into running_cs, so
        // the checksum falls out for free — no second pass over the bytes.
        while (p < end) {
            const unsigned cs_before_field = running_cs;  // checksum excludes tag-10 field

            // --- tag: digits up to '=' (FIX tags are always numeric) ---
            unsigned tag = 0;
            char c;
            while (p < end && (c = *p) != '=') {
                running_cs += static_cast<unsigned char>(c);
                tag = tag * 10 + static_cast<unsigned>(c - '0');
                ++p;
            }
            if (p == end) break;                       // truncated field
            running_cs += static_cast<unsigned char>('=');
            ++p;                                       // skip '='

            // --- value: bytes up to SOH ---
            // Hottest loop in the parser (runs over every value byte). Well-formed
            // FIX guarantees an SOH terminator before end-of-buffer, so we trust
            // that sentinel and drop the redundant `p < end` bounds check — one
            // fewer branch per byte. (Outer and tag loops stay bounded.)
            const char* val_start = p;
            while ((c = *p) != '\x01') {
                running_cs += static_cast<unsigned char>(c);
                ++p;
            }
            std::string_view value(val_start, static_cast<size_t>(p - val_start));
            running_cs += static_cast<unsigned char>('\x01');
            ++p;                                       // consume SOH

            if (tag == 10) {
                // Expected checksum is this field's value; actual is the sum of
                // everything before the "10=" field started.
                unsigned expected = 0;
                for (char d : value) expected = expected * 10 + static_cast<unsigned>(d - '0');
                order.valid = ((cs_before_field & 0xFFu) == expected);
                msg_complete = true;
                break;
            }

            switch (tag) {
                case 35: if (!value.empty()) order.msg_type = value[0]; break;
                case 54: if (!value.empty()) order.side = static_cast<int8_t>(value[0] - '0'); break;
                case 55: lookup(value, order.symbol_id); break;
                case 52: order.timestamp = parse_timestamp(value); break;
                case 44: order.price = parse_price(value); break;
                case 38: order.quantity = parse_int(value); break;
                default: break;
            }
        }

        if (!msg_complete) break;   // no more complete messages
        out.push_back(order);
    }
}

} // namespace hftu
