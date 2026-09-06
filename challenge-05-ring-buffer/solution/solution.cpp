#include "solution.h"

namespace hftu {
    RingBuffer::RingBuffer(size_t capacity)
        : buf_(capacity), capacity_(capacity) {
        kMask = capacity_ - 1;
    }

    bool RingBuffer::push(const Message &msg) {
        if (head_ == cached_tail_ + capacity_) {
            cached_tail_ = tail_.load(std::memory_order::seq_cst);
            if (head_ == cached_tail_ + capacity_) {
                return false;
            }
        }
        buf_[head_ & kMask] = msg;
        head_ = head_ + 1;
        return true;
    }

    bool RingBuffer::pop(Message &out) {
        if (tail_ == cached_head_) {
            cached_head_ = head_.load(std::memory_order::seq_cst);
            if (tail_ == cached_head_) {
                return false;
            }
        }
        out = buf_[tail_ & kMask];
        tail_ = tail_ + 1;
        return true;
    }

    size_t RingBuffer::size() const {
        return head_ - tail_;
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
