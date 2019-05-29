#ifndef ECVL_ITERATORS_H_
#define ECVL_ITERATORS_H_

#include <vector>
#include <cstdint>

namespace ecvl {

class Image;

template <typename T>
struct Iterator {
    std::vector<int> pos_;
    uint8_t* ptr_;
    Image* img_;

    typedef Iterator& (Iterator::*IncrementMemFn)();
    IncrementMemFn incrementor = &Iterator<T>::IncrementPos;

    Iterator(Image& img, std::vector<int> pos = {});
    Iterator& operator++() /* prefix */ { return (this->*incrementor)(); }
    T& operator* () const { return *reinterpret_cast<T*>(ptr_); }
    T* operator-> () const { return reinterpret_cast<T*>(ptr_); }
    bool operator==(const Iterator& rhs) const { return ptr_ == rhs.ptr_; }
    bool operator!=(const Iterator& rhs) const { return ptr_ != rhs.ptr_; }
private:
    Iterator & IncrementPos();
    Iterator& ContiguousIncrementPos();
};

template <typename T>
struct ConstIterator {
    std::vector<int> pos_;
    const uint8_t* ptr_;
    const Image* img_;

    typedef ConstIterator& (ConstIterator::*IncrementMemFn)();
    IncrementMemFn incrementor = &ConstIterator<T>::IncrementPos;

    ConstIterator(const Image& img, std::vector<int> pos = {});
    ConstIterator& operator++() /* prefix */ { return (this->*incrementor)(); }
    const T& operator* () const { return *reinterpret_cast<const T*>(ptr_); }
    const T* operator-> () const { return reinterpret_cast<const T*>(ptr_); }
    bool operator==(const ConstIterator& rhs) const { return ptr_ == rhs.ptr_; }
    bool operator!=(const ConstIterator& rhs) const { return ptr_ != rhs.ptr_; }
private:
    ConstIterator & IncrementPos();
    ConstIterator& ContiguousIncrementPos();
};


template <typename T>
struct ContiguousIterator {
    uint8_t* ptr_;
    Image* img_;

    ContiguousIterator(Image& img, std::vector<int> pos = {});
    ContiguousIterator& operator++() /* prefix */ { return ContiguousIncrementPos(); }
    T& operator* () const { return *reinterpret_cast<T*>(ptr_); }
    T* operator-> () const { return reinterpret_cast<T*>(ptr_); }
    bool operator==(const ContiguousIterator& rhs) const { return ptr_ == rhs.ptr_; }
    bool operator!=(const ContiguousIterator& rhs) const { return ptr_ != rhs.ptr_; }
private:
    ContiguousIterator& ContiguousIncrementPos();
};

template <typename T>
struct ConstContiguousIterator {
    uint8_t* ptr_;
    const Image* img_;

    ConstContiguousIterator(const Image& img, std::vector<int> pos = {});
    ConstContiguousIterator& operator++() /* prefix */ { return ContiguousIncrementPos(); }
    const T& operator* () const { return *reinterpret_cast<const T*>(ptr_); }
    const T* operator-> () const { return reinterpret_cast<const T*>(ptr_); }
    bool operator==(const ConstContiguousIterator& rhs) const { return ptr_ == rhs.ptr_; }
    bool operator!=(const ConstContiguousIterator& rhs) const { return ptr_ != rhs.ptr_; }
private:
    ConstContiguousIterator & ContiguousIncrementPos();
};

} // namespace ecvl

#endif // !ECVL_ITERATORS_H_
