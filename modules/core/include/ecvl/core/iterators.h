/*
* ECVL - European Computer Vision Library
* Version: 1.0.0
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#ifndef ECVL_ITERATORS_H_
#define ECVL_ITERATORS_H_

#include <vector>
#include <cstdint>

namespace ecvl
{
class Image;

template <typename T>
struct Iterator
{
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    std::vector<int> pos_;
    uint8_t* ptr_;
    Image* img_;

    typedef Iterator& (Iterator::* IncrementMemFn)();
    IncrementMemFn incrementor = &Iterator<T>::IncrementPos;

    Iterator() {}
    Iterator(Image& img, std::vector<int> pos = {});
    Iterator& operator++() /* prefix */ { return (this->*incrementor)(); }
    T& operator* () const { return *reinterpret_cast<T*>(ptr_); }
    T* operator-> () const { return reinterpret_cast<T*>(ptr_); }
    bool operator==(const Iterator& rhs) const { return ptr_ == rhs.ptr_; }
    bool operator!=(const Iterator& rhs) const { return ptr_ != rhs.ptr_; }
private:
    Iterator& IncrementPos();
    Iterator& ContiguousIncrementPos();
};

template <typename T>
struct ConstIterator
{
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = const value_type*;
    using reference = const value_type&;
    using iterator_category = std::forward_iterator_tag;

    std::vector<int> pos_;
    const uint8_t* ptr_;
    const Image* img_;

    typedef ConstIterator& (ConstIterator::* IncrementMemFn)();
    IncrementMemFn incrementor = &ConstIterator<T>::IncrementPos;

    ConstIterator() {}
    ConstIterator(const Image& img, std::vector<int> pos = {});
    ConstIterator& operator++() /* prefix */ { return (this->*incrementor)(); }
    const T& operator* () const { return *reinterpret_cast<const T*>(ptr_); }
    const T* operator-> () const { return reinterpret_cast<const T*>(ptr_); }
    bool operator==(const ConstIterator& rhs) const { return ptr_ == rhs.ptr_; }
    bool operator!=(const ConstIterator& rhs) const { return ptr_ != rhs.ptr_; }
private:
    ConstIterator& IncrementPos();
    ConstIterator& ContiguousIncrementPos();
};

template <typename T>
struct ContiguousIterator
{
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    uint8_t* ptr_;
    Image* img_;

    ContiguousIterator() {}
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
struct ConstContiguousIterator
{
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = const value_type*;
    using reference = const value_type&;
    using iterator_category = std::forward_iterator_tag;

    const uint8_t* ptr_;
    const Image* img_;

    ConstContiguousIterator() {}
    ConstContiguousIterator(const Image& img, std::vector<int> pos = {});
    ConstContiguousIterator& operator++() /* prefix */ { return ContiguousIncrementPos(); }
    const T& operator* () const { return *reinterpret_cast<const T*>(ptr_); }
    const T* operator-> () const { return reinterpret_cast<const T*>(ptr_); }
    bool operator==(const ConstContiguousIterator& rhs) const { return ptr_ == rhs.ptr_; }
    bool operator!=(const ConstContiguousIterator& rhs) const { return ptr_ != rhs.ptr_; }
private:
    ConstContiguousIterator& ContiguousIncrementPos();
};

/** @example example_core_iterators.cpp
 Iterators example.
*/
} // namespace ecvl

#endif // !ECVL_ITERATORS_H_
