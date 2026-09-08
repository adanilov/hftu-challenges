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

// Scan [begin, end): return sum of all bytes (pre-mod checksum) and append the
// offset (relative to begin) of the byte AFTER each SOH — i.e. each field start
// — to starts[]. begin itself (the first field) is added by the caller.
// Returns the number of starts written.
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
        uint8x16_t cmp = vceqq_u8(v, soh);
        uint8x16_t sel = vandq_u8(cmp, bits);
        unsigned m = vaddv_u8(vget_low_u8(sel)) | (vaddv_u8(vget_high_u8(sel)) << 8);
        while (m) {
            int b = __builtin_ctz(m);
            starts[n++] = p + b + 1;
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
            int b = __builtin_ctz(m);
            starts[n++] = p + b + 1;
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

// SOH-anchored 3-gram of a 2-digit tag: bytes "d d =" packed little-endian.
static constexpr uint32_t sig3(char a, char b) {
    return static_cast<uint32_t>(static_cast<unsigned char>(a))
         | (static_cast<uint32_t>(static_cast<unsigned char>(b)) << 8)
         | (static_cast<uint32_t>(static_cast<unsigned char>('=')) << 16);
}

void FixParser::parse_batch(std::string_view data, std::vector<ParsedOrder>& out) {
    const char* const end = data.data() + data.size();
    const char* p = data.data();

    constexpr uint32_t SIG_35 = sig3('3','5');   // MsgType
    constexpr uint32_t SIG_52 = sig3('5','2');   // SendingTime
    constexpr uint32_t SIG_55 = sig3('5','5');   // Symbol
    constexpr uint32_t SIG_44 = sig3('4','4');   // Price
    constexpr uint32_t SIG_38 = sig3('3','8');   // OrderQty
    constexpr uint32_t SIG_54 = sig3('5','4');   // Side

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

        // --- One fused SIMD pass: checksum + every field boundary (SOH). ---
        const char* starts[512];
        starts[0] = p;                             // first field (tag 8)
        unsigned csum = 0;
        int ns = 1 + scan_region(p, tag10_start, starts + 1, csum);

        // --- Dispatch the 6 fields we care about via 3-gram signature. ---
        ParsedOrder order{};
        for (int k = 0; k + 1 < ns; ++k) {
            const char* s = starts[k];
            const char* e = starts[k + 1] - 1;     // SOH terminating this field
            uint32_t sig;
            std::memcpy(&sig, s, 4);
            sig &= 0x00FFFFFFu;                    // low 3 bytes = "dd="
            const std::string_view value(s + 3, static_cast<size_t>(e - (s + 3)));

            if      (sig == SIG_55) lookup(value, order.symbol_id);
            else if (sig == SIG_44) order.price = parse_price(value);
            else if (sig == SIG_38) order.quantity = parse_int(value);
            else if (sig == SIG_52) order.timestamp = parse_timestamp(value);
            else if (sig == SIG_35) order.msg_type = s[3];
            else if (sig == SIG_54) order.side = static_cast<int8_t>(s[3] - '0');
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
