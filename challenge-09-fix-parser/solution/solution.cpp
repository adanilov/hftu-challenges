// Challenge 09: FIX Parser
//
// Hot path design:
//   * Message framing is arithmetic, not scanning: tag 9 (BodyLength) plus the
//     fixed 7-byte "10=NNN<SOH>" checksum field give the exact message extent,
//     so we never scan for message boundaries.
//   * One fused SIMD pass over the checksum region does BOTH jobs from the same
//     16-byte loads: it accumulates the checksum (sum of all bytes) and finds
//     every SOH (field boundary). No byte is scanned twice.
//   * Fields are dispatched by an SOH-anchored 3-gram: the 3 bytes "dd=" at a
//     field start uniquely identify a 2-digit tag (anchoring on the SOH avoids
//     the 10-vs-100 substring ambiguity). Filler fields never match, so their
//     bytes cost only the (already-free) checksum add — no per-field parsing.

#include "solution.h"

#include <array>
#include <cstring>   // memcpy

#if defined(FIX_FORCE_SCALAR)
  // no SIMD
#elif defined(__aarch64__)
  #include <arm_neon.h>
  #define FIX_SIMD_NEON 1
#elif defined(__SSE2__)
  #include <emmintrin.h>
  #define FIX_SIMD_SSE2 1
#endif

namespace hftu {

// Scan [begin, end): return the sum of all bytes (pre-mod checksum) and append
// each field start (the byte after every SOH) to starts[]. Kept separate from
// dispatch on purpose — decoupling lets the SIMD scan run at full throughput;
// fusing the dispatch into this loop measured slower (it bloats the hot loop).
static int scan_region(const char* begin, const char* end,
                       const char** starts, unsigned& csum_out) {
    unsigned csum = 0;
    int n = 0;
    const char* p = begin;

#if defined(FIX_SIMD_NEON)
    const uint8x16_t soh = vdupq_n_u8(0x01);
    // Per-byte bit selector to turn a 0x00/0xFF compare mask into a 16-bit mask.
    const uint8x16_t bits = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    for (; p + 16 <= end; p += 16) {
        uint8x16_t v = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
        csum += vaddlvq_u8(v);                       // widening sum of 16 bytes
        uint8x16_t sel = vandq_u8(vceqq_u8(v, soh), bits);
        unsigned m = vaddv_u8(vget_low_u8(sel)) | (vaddv_u8(vget_high_u8(sel)) << 8);
        while (m) {
            starts[n++] = p + __builtin_ctz(m) + 1;
            m &= m - 1;
        }
    }
#elif defined(FIX_SIMD_SSE2)
    const __m128i soh  = _mm_set1_epi8(0x01);
    const __m128i zero = _mm_setzero_si128();
    for (; p + 16 <= end; p += 16) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        __m128i sad = _mm_sad_epu8(v, zero);         // two 64-bit partial sums
        csum += static_cast<unsigned>(_mm_cvtsi128_si32(sad))
              + static_cast<unsigned>(_mm_extract_epi16(sad, 4));
        unsigned m = static_cast<unsigned>(
            _mm_movemask_epi8(_mm_cmpeq_epi8(v, soh)));
        while (m) {
            starts[n++] = p + __builtin_ctz(m) + 1;
            m &= m - 1;
        }
    }
#endif

    // Scalar tail (and the whole region on non-SIMD builds).
    for (; p < end; ++p) {
        unsigned char c = static_cast<unsigned char>(*p);
        csum += c;
        if (c == 0x01) starts[n++] = p + 1;
    }
    csum_out = csum;
    return n;
}

FixParser::FixParser() {}

void FixParser::build(std::span<const TickerEntry> entries) {
    // Size the fast table to the next power of two >= 2*N (short probe chains).
    size_t cap = 16;
    while (cap < entries.size() * 2) cap <<= 1;
    fast_.assign(cap, Slot{});
    fast_mask_ = cap - 1;

    for (const auto& e : entries) {
        if (e.symbol.size() <= 8) {
            uint64_t k = pack8(e.symbol.data(), e.symbol.size());
            size_t idx = hash64(k) & fast_mask_;
            while (fast_[idx].used) idx = (idx + 1) & fast_mask_;   // linear probe
            fast_[idx] = Slot{k, e.value, 1};
        } else {
            long_.emplace_back(std::string(e.symbol), e.value);
        }
    }
}

// Keep only the low `len` bytes of an 8-byte load (len in 0..8).
static constexpr uint64_t LOW_BYTES[9] = {
    0x0000000000000000ull, 0x00000000000000FFull, 0x000000000000FFFFull,
    0x0000000000FFFFFFull, 0x00000000FFFFFFFFull, 0x000000FFFFFFFFFFull,
    0x0000FFFFFFFFFFFFull, 0x00FFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull
};

bool FixParser::lookup(std::string_view s, uint32_t& out) const {
    const size_t L = s.size();
    if (L <= 8) {
        // Safe 8-byte over-read: the symbol is never the last field (tag 10 is),
        // so >= 8 readable bytes always follow; the extra bytes are masked off.
        uint64_t k;
        std::memcpy(&k, s.data(), 8);
        k &= LOW_BYTES[L];
        size_t idx = hash64(k) & fast_mask_;
        while (fast_[idx].used) {
            if (fast_[idx].key == k) { out = fast_[idx].val; return true; }
            idx = (idx + 1) & fast_mask_;
        }
        return false;
    }
    for (const auto& pr : long_)
        if (pr.first == s) { out = pr.second; return true; }
    return false;
}

static constexpr int64_t POW10[] = {
    1LL, 10LL, 100LL, 1'000LL, 10'000LL, 100'000LL, 1'000'000LL,
    10'000'000LL, 100'000'000LL, 1'000'000'000LL
};

// SWAR: parse exactly 8 ASCII digits (p[0] most significant) in a handful of
// multiply/shift ops instead of a digit loop. Little-endian only (x86 + Apple
// Silicon both qualify). Standard "parse eight digits" bit-trick.
static inline uint32_t parse_8digits(const char* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    v = (v & 0x0F0F0F0F0F0F0F0Full) * 2561 >> 8;
    v = (v & 0x00FF00FF00FF00FFull) * 6553601 >> 16;
    v = (v & 0x0000FFFF0000FFFFull) * 42949672960001ull >> 32;
    return static_cast<uint32_t>(v);
}

// Parse a fixed-point decimal string like "187.50" into int64_t * 10^8.
static int64_t parse_price(std::string_view s) {
    int64_t whole = 0, frac = 0;
    int frac_digits = 0;
    bool in_frac = false;
    for (char c : s) {
        if (c == '.') { in_frac = true; continue; }
        unsigned d = static_cast<unsigned>(c - '0');
        if (in_frac) { frac = frac * 10 + d; ++frac_digits; }
        else         { whole = whole * 10 + d; }
    }
    // Scale the fraction to 8 decimals via a table lookup (no scaling loop).
    if (frac_digits <= 8) frac *= POW10[8 - frac_digits];
    else                  for (int i = 8; i < frac_digits; ++i) frac /= 10;
    return whole * 100'000'000LL + frac;
}

static int64_t parse_int(std::string_view s) {
    int64_t val = 0;
    for (char c : s) val = val * 10 + (c - '0');   // value is all digits
    return val;
}

// Parse FIX timestamp "YYYYMMDD-HH:MM:SS.nnnnnnnnn" to nanoseconds from epoch
static int64_t parse_timestamp(std::string_view s) {
    // Minimal parsing: extract components
    // Format: 20260320-14:30:00.123456789
    if (s.size() < 17) return 0;

    const char* p = s.data();
    int64_t nanos = 0;

    // YYYYMMDD is 8 contiguous digits at a fixed offset → one SWAR parse.
    uint32_t ymd = parse_8digits(p);
    int year  = static_cast<int>(ymd / 10000);
    int month = static_cast<int>((ymd / 100) % 100);
    int day   = static_cast<int>(ymd % 100);
    // HH:MM:SS at fixed offsets (colons at 8/11/14 skipped).
    int hour = (p[9]  - '0') * 10 + (p[10] - '0');
    int min  = (p[12] - '0') * 10 + (p[13] - '0');
    int sec  = (p[15] - '0') * 10 + (p[16] - '0');

    // Fractional seconds (variable length), scaled to nanoseconds via table.
    if (s.size() > 17 && p[17] == '.') {
        int frac_digits = 0;
        for (size_t i = 18; i < s.size() && p[i] >= '0' && p[i] <= '9'; ++i) {
            nanos = nanos * 10 + (p[i] - '0');
            ++frac_digits;
        }
        if (frac_digits <= 9) nanos *= POW10[9 - frac_digits];
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

// Maps a 2-digit tag (0..99) to a dense handler id (0 = not wanted).
static constexpr std::array<uint8_t, 100> TAG_ACTION = [] {
    std::array<uint8_t, 100> a{};
    a[55] = 1; a[44] = 2; a[38] = 3; a[52] = 4; a[35] = 5; a[54] = 6;
    return a;
}();

void FixParser::parse_batch(std::string_view data, std::vector<ParsedOrder>& out) {
    const char* const end = data.data() + data.size();
    const char* p = data.data();

    while (p < end) {
        // --- Frame the message by arithmetic (tags 8 and 9), no scanning. ---
        // Tag 8 (BeginString) is first: skip to its SOH.
        const char* soh8 = static_cast<const char*>(
            memchr(p, 0x01, static_cast<size_t>(end - p)));
        if (!soh8) break;
        // Tag 9 (BodyLength) is second: "9=<N>". Read N.
        const char* np = soh8 + 3;                 // skip SOH + "9="
        unsigned N = 0;
        while (*np != 0x01) { N = N * 10 + static_cast<unsigned>(*np - '0'); ++np; }
        const char* const header_end = np + 1;     // first body field
        const char* const tag10_start = header_end + N;   // "10=NNN<SOH>"

        // --- SIMD scan: checksum + field starts. ---
        const char* starts[512];
        starts[0] = p;                             // first field (tag 8)
        unsigned csum = 0;
        int ns = 1 + scan_region(p, tag10_start, starts + 1, csum);

        // --- Dispatch. Two well-predicted reject branches drop the ~18 filler
        //     fields per message (non-2-digit tag, or tag with no handler);
        //     only the ~6 real fields reach the switch. ---
        ParsedOrder order{};
        for (int k = 0; k + 1 < ns; ++k) {
            const char* s = starts[k];
            if (s[2] != '=') continue;             // not a 2-digit tag
            unsigned tag = static_cast<unsigned>(s[0] - '0') * 10u
                         + static_cast<unsigned>(s[1] - '0');
            uint8_t act = TAG_ACTION[tag];
            if (!act) continue;                    // 2-digit tag we don't want
            const char* e = starts[k + 1] - 1;
            std::string_view value(s + 3, static_cast<size_t>(e - s - 3));
            switch (act) {
                case 1: lookup(value, order.symbol_id); break;         // 55
                case 2: order.price = parse_price(value); break;       // 44
                case 3: order.quantity = parse_int(value); break;      // 38
                case 4: order.timestamp = parse_timestamp(value); break; // 52
                case 5: order.msg_type = s[3]; break;                  // 35
                case 6: order.side = static_cast<int8_t>(s[3] - '0'); break; // 54
            }
        }

        // --- Checksum: "10=NNN" is 3 zero-padded digits. ---
        const unsigned expected = static_cast<unsigned>(tag10_start[3] - '0') * 100u
                                + static_cast<unsigned>(tag10_start[4] - '0') * 10u
                                + static_cast<unsigned>(tag10_start[5] - '0');
        order.valid = ((csum & 0xFFu) == expected);

        out.push_back(order);
        p = tag10_start + 7;                       // "10=" + 3 digits + SOH
    }
}

} // namespace hftu
