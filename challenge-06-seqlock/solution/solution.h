#pragma once
// Challenge 06: Seqlock
//
// A seqlock allows one writer and multiple readers to share data
// without blocking. Readers retry if they detect a concurrent write.
// No mutexes — this is a lock-free exercise.
//
// read()/write() are defined inline in the header so they inline into the
// caller's hot loop (Ch 15): removes the call/ret per op and lets the compiler
// keep &seq_ in a register, fuse the retry loop, and vectorize the copy.

#include <atomic>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  #define HFTU_CPU_RELAX() _mm_pause()   // helps under contention on x86
#else
  #define HFTU_CPU_RELAX() ((void)0)     // measured net-negative on Apple Silicon
#endif

namespace hftu {

    struct Payload {
        int64_t a;
        int64_t b;
        int64_t c;
        int64_t d;
    };

    class Seqlock {
        alignas(64) std::atomic<uint32_t> seq_{0};
        Payload data_{};
    public:
        Seqlock() = default;

        // Update the protected payload. Single writer.
        inline void write(const Payload& data) {
            uint32_t s = seq_.load(std::memory_order_relaxed);
            seq_.store(s + 1, std::memory_order_relaxed);   // -> odd: write in progress
            std::atomic_thread_fence(std::memory_order_release);
            data_ = data;                                    // whole-struct copy -> ldp/stp
            seq_.store(s + 2, std::memory_order_release);    // -> even: publish
        }

        // Read the payload. Must never return a torn value.
        inline Payload read() const {
            Payload out;
            for (;;) {
                uint32_t s0 = seq_.load(std::memory_order_acquire);
                if (s0 & 1) {                    // write in progress: skip the copy
                    HFTU_CPU_RELAX();
                    continue;
                }
                out = data_;
                std::atomic_thread_fence(std::memory_order_acquire);
                if (seq_.load(std::memory_order_relaxed) == s0) [[likely]]
                    return out;                  // version unchanged & even -> clean read
                HFTU_CPU_RELAX();                // collided with a write, retry
            }
        }
    };

} // namespace hftu
