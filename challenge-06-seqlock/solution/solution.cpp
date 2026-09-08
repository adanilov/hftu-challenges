// Challenge 06: Seqlock — Skeleton Implementation
// This is a correct but slow mutex-based reference. You can do MUCH better!
// The real seqlock uses a sequence counter and memory fences — no mutexes.

#include "solution.h"

namespace hftu {
    Seqlock::Seqlock() {
    }

    void Seqlock::write(const Payload &data) {
        uint32_t s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        data_.a = data.a;
        data_.b = data.b;
        data_.c = data.c;
        data_.d = data.d;
        seq_.store(s + 2, std::memory_order_release);
    }

    Payload Seqlock::read() const {
        Payload out;
        uint32_t s0, s1;
        do {
            s0 = seq_.load(std::memory_order_acquire);
            out = data_;
            std::atomic_thread_fence(std::memory_order_acquire);
            s1 = seq_.load(std::memory_order_relaxed);
        } while (s0 != s1 || (s0 & 1));
        return out;
    }
} // namespace hftu
