// Challenge 06: Seqlock
// Sequence-counter seqlock: writer never blocks, readers retry on conflict.
// Optimization over the canonical reference: the reader checks for an
// in-progress write (odd version) BEFORE copying the payload, so a losing
// iteration costs one load + branch instead of load + 32B copy + fence + load.

#include "solution.h"

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  #define CPU_RELAX() _mm_pause()   // helps under contention on x86 (certified target)
#else
  #define CPU_RELAX() ((void)0)     // measured net-negative on Apple Silicon
#endif

namespace hftu {
    Seqlock::Seqlock() {
    }

    void Seqlock::write(const Payload &data) {
        uint32_t s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1, std::memory_order_relaxed);   // -> odd: write in progress
        std::atomic_thread_fence(std::memory_order_release);
        data_.a = data.a;
        data_.b = data.b;
        data_.c = data.c;
        data_.d = data.d;
        seq_.store(s + 2, std::memory_order_release);    // -> even: publish
    }

    Payload Seqlock::read() const {
        Payload out;
        for (;;) {
            uint32_t s0 = seq_.load(std::memory_order_acquire);
            if (s0 & 1) {                    // write in progress: skip the copy
                CPU_RELAX();
                continue;
            }
            out = data_;
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq_.load(std::memory_order_relaxed) == s0) [[likely]]
                return out;                  // version unchanged & even -> clean read
            CPU_RELAX();                     // collided with a write, retry
        }
    }
} // namespace hftu
