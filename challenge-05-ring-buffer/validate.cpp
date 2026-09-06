// Correctness check for Challenge 05: Ring Buffer (SPSC)
// Registered via RegisterValidation, so run_benchmarks() runs it first and
// refuses to benchmark an incorrect solution (see common/benchmark_harness.h).

#include "common/benchmark_harness.h"
#include "solution/solution.h"

#include <thread>
#include <atomic>

namespace {

hftu::Message make_msg(uint64_t i) {
    hftu::Message m{};
    m.timestamp  = i * 7 + 1;
    m.symbol_id  = static_cast<uint32_t>(i & 0xFFF);
    m.side       = static_cast<uint16_t>(i & 1);
    m.flags      = static_cast<uint16_t>((i >> 1) & 1);
    m.price      = static_cast<int64_t>(i) * 100 + 1;
    m.quantity   = static_cast<int64_t>((i & 0xFF) + 1);
    m.order_id   = static_cast<int64_t>(i);
    m.sequence   = i;
    return m;
}

bool same(const hftu::Message& a, const hftu::Message& b) {
    return a.timestamp == b.timestamp && a.symbol_id == b.symbol_id &&
           a.side == b.side && a.flags == b.flags && a.price == b.price &&
           a.quantity == b.quantity && a.order_id == b.order_id &&
           a.sequence == b.sequence;
}

// Single-threaded FIFO ordering, payload integrity, and full/empty behaviour.
bool validate_single_thread() {
    const size_t cap = 1024;
    hftu::RingBuffer rb(cap);

    hftu::Message out{};
    if (rb.pop(out)) return hftu::check_failed("single_thread", "pop on empty returned true");
    if (rb.size() != 0) return hftu::check_failed("single_thread", "size not 0 when empty");

    // Fill to capacity, then push must fail.
    for (size_t i = 0; i < cap; ++i) {
        if (!rb.push(make_msg(i))) return hftu::check_failed("single_thread", "push failed before full");
    }
    if (rb.size() != cap) return hftu::check_failed("single_thread", "size wrong when full");
    if (rb.push(make_msg(999999))) return hftu::check_failed("single_thread", "push on full returned true");

    // Drain: must come back in FIFO order with identical payloads.
    for (size_t i = 0; i < cap; ++i) {
        if (!rb.pop(out)) return hftu::check_failed("single_thread", "pop failed before empty");
        if (!same(out, make_msg(i))) return hftu::check_failed("single_thread", "FIFO order / payload mismatch");
    }
    if (rb.pop(out)) return hftu::check_failed("single_thread", "pop on empty returned true after drain");

    return true;
}

// Concurrent producer/consumer: every message must arrive exactly once, in
// order, with an intact payload. Wraps around the buffer many times.
bool validate_concurrent() {
    const size_t cap = 1024;
    const uint64_t total = 1'000'000;
    hftu::RingBuffer rb(cap);
    std::atomic<bool> failed{false};

    std::thread consumer([&]() {
        hftu::Message out{};
        for (uint64_t i = 0; i < total; ++i) {
            while (!rb.pop(out)) {
                if (failed.load(std::memory_order_relaxed)) return;
            }
            if (!same(out, make_msg(i))) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
        }
    });

    std::thread producer([&]() {
        for (uint64_t i = 0; i < total; ++i) {
            while (!rb.push(make_msg(i))) {
                if (failed.load(std::memory_order_relaxed)) return;
            }
        }
    });

    producer.join();
    consumer.join();

    if (failed.load()) return hftu::check_failed("concurrent", "out-of-order, dropped, or corrupted message");
    return true;
}

} // namespace

static hftu::RegisterValidation reg_single("ring_buffer_single_thread", validate_single_thread);
static hftu::RegisterValidation reg_concurrent("ring_buffer_concurrent", validate_concurrent);
