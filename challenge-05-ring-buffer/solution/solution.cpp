#include "solution.h"

namespace hftu {
    RingBuffer::RingBuffer(size_t capacity)
        : buf_(capacity), capacity_(capacity) {
        kMask = capacity_ - 1;
    }

    bool RingBuffer::push(const Message &msg) {
        size_t head = head_.load(std::memory_order::relaxed);
        if (head == cached_tail_ + capacity_) [[unlikely]] {   // full: rare in a balanced queue
            cached_tail_ = tail_.load(std::memory_order::acquire);
            if (head == cached_tail_ + capacity_) {
                return false;
            }
        }
        buf_[head & kMask] = msg;
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    bool RingBuffer::pop(Message &out) {
        size_t tail = tail_.load(std::memory_order::relaxed);
        if (tail == cached_head_) [[unlikely]] {   // empty: rare in a balanced queue
            cached_head_ = head_.load(std::memory_order::acquire);
            if (tail == cached_head_) {
                return false;
            }
        }
        out = buf_[tail & kMask];
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    size_t RingBuffer::size() const {
        return head_.load(std::memory_order::relaxed) - tail_.load(std::memory_order::relaxed);
    }
} // namespace hftu
// int main() {
// hftu::RingBuffer rb(8);
// rb.push(hftu::Message());
// std::cout << "size=" << rb.size()<<std::endl;
// hftu::Message m;
// bool pop = rb.pop(m);
// std::cout << "size=" << rb.size()<<std::endl;
// std::cout << pop << std::endl;
// }
