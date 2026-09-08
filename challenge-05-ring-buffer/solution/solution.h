#pragma once
// Challenge 05: Ring Buffer (SPSC)
// Edit this file and solution.cpp to implement your solution.
//
// This is a Single-Producer Single-Consumer ring buffer.
// The producer calls push() from one thread, the consumer calls pop() from another.
// You MUST ensure thread safety between the producer and consumer.

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <vector>

namespace hftu {

// 48-byte market data message — realistic size, not power-of-2 aligned.
struct Message {
    uint64_t timestamp;   // 8
    uint32_t symbol_id;   // 4
    uint16_t side;        // 2
    uint16_t flags;       // 2
    int64_t  price;       // 8
    int64_t  quantity;    // 8
    int64_t  order_id;   // 8
    uint64_t sequence;    // 8
};                        // 48 bytes total

class RingBuffer {
public:
    // Capacity is always a power of 2.
    // Defined inline in the header so push/pop inline into the caller's hot loop
    // (Ch 15): no call/ret per op, indices kept in registers, loop fused.
    explicit RingBuffer(size_t capacity)
        : buf_(capacity), capacity_(capacity) {
        kMask = capacity_ - 1;
    }

    // Push a message (producer thread). Returns false if full.
    inline bool push(const Message& msg) {
        size_t head = head_.load(std::memory_order::relaxed);
        if (head == cached_tail_ + capacity_) {
            cached_tail_ = tail_.load(std::memory_order::acquire);
            if (head == cached_tail_ + capacity_) {
                return false;
            }
        }
        buf_[head & kMask] = msg;
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    // Pop a message into out (consumer thread). Returns false if empty.
    inline bool pop(Message& out) {
        size_t tail = tail_.load(std::memory_order::relaxed);
        if (tail == cached_head_) {
            cached_head_ = head_.load(std::memory_order::acquire);
            if (tail == cached_head_) {
                return false;
            }
        }
        out = buf_[tail & kMask];
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    // Number of elements currently stored.
    inline size_t size() const {
        return head_.load(std::memory_order::relaxed) - tail_.load(std::memory_order::relaxed);
    }

private:
    std::vector<Message> buf_;
    size_t capacity_;
    alignas(64) std::atomic<size_t> head_ = 0;
    // std::atomic<size_t> cached_tail_ = 0;
    size_t cached_tail_ = 0;

    alignas(64) std::atomic<size_t> tail_ = 0;
    // std::atomic<size_t> cached_head_ = 0;
    size_t cached_head_ = 0;
    int kMask;
};

} // namespace hftu
