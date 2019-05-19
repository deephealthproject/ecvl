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
    IncrementMemFn incrementor = &Iterator::IncrementPos;

    explicit Iterator(Image& img, std::vector<int> pos = {}) : img_{ &img }, pos_{ move(pos) }
    {
        if (img_->contiguous_)
            incrementor = &Iterator::ContiguousIncrementPos;
        if (pos_.empty()) { // Begin
            pos_.resize(img_->dims_.size(), 0);
            ptr_ = img.data_;
        }
        else {
            if (pos_.size() != img_->dims_.size()) {
                throw std::runtime_error("Iterator starting pos has a wrong size");
            }
            if (pos_ == img_->dims_) { // End 
                if (img_->contiguous_) {
                    ptr_ = img_->data_ + img_->datasize_;
                }
                else {
                    ptr_ = nullptr;
                }
            }
            else {
                ptr_ = img_->Ptr(pos_);
            }
        }
    }
    Iterator& operator++() /* prefix */ { return (this->*incrementor)(); }
    T& operator* () const { return *reinterpret_cast<T*>(ptr_); }
    T* operator-> () const { return reinterpret_cast<T*>(ptr_); }
    bool operator==(const Iterator& rhs) const { return ptr_ == rhs.ptr_; }
    bool operator!=(const Iterator& rhs) const { return ptr_ != rhs.ptr_; }
private:
    Iterator& IncrementPos()
    {
        int spos = pos_.size();
        int dim;
        for (dim = 0; dim < spos; ++dim) {
            ++pos_[dim];
            ptr_ += img_->strides_[dim];
            if (pos_[dim] != img_->dims_[dim])
                break;
            // Back to dimension starting position
            pos_[dim] = 0;
            ptr_ -= img_->dims_[dim] * img_->strides_[dim];
        }
        if (dim == spos)
            ptr_ = nullptr;
        return *this;
    }
    Iterator& ContiguousIncrementPos()
    {
        ptr_ += img_->elemsize_;
        return *this;
    }
};

template <typename T>
struct ConstIterator {
    std::vector<int> pos_;
    uint8_t* ptr_;
    const Image* img_;

    typedef ConstIterator& (ConstIterator::*IncrementMemFn)();
    IncrementMemFn incrementor = &Iterator::IncrementPos;

    explicit ConstIterator(const Image& img, std::vector<int> pos = {}) : img_{ &img }, pos_{ move(pos) }
    {
        if (img_->contiguous_)
            incrementor = &ConstIterator::ContiguousIncrementPos;
        if (pos_.empty()) { // Begin
            pos_.resize(img_->dims_.size(), 0);
            ptr_ = img.data_;
        }
        else {
            if (pos_.size() != img_->dims_.size()) {
                throw std::runtime_error("ConstIterator starting pos has a wrong size");
            }
            if (pos_ == img_->dims_) { // End 
                if (img_->contiguous_) {
                    ptr_ = img_->data_ + img_->datasize_;
                }
                else {
                    ptr_ = nullptr;
                }
            }
            else {
                ptr_ = img_->Ptr(pos_);
            }
        }
    }
    ConstIterator& operator++() /* prefix */ { return (this->*incrementor)(); }
    const T& operator* () const { return *reinterpret_cast<const T*>(ptr_); }
    const T* operator-> () const { return reinterpret_cast<const T*>(ptr_); }
    bool operator==(const ConstIterator& rhs) const { return ptr_ == rhs.ptr_; }
    bool operator!=(const ConstIterator& rhs) const { return ptr_ != rhs.ptr_; }
private:
    ConstIterator& IncrementPos()
    {
        int spos = pos_.size();
        int dim;
        for (dim = 0; dim < spos; ++dim) {
            ++pos_[dim];
            ptr_ += img_->strides_[dim];
            if (pos_[dim] != img_->dims_[dim])
                break;
            // Back to dimension starting position
            pos_[dim] = 0;
            ptr_ -= img_->dims_[dim] * img_->strides_[dim];
        }
        if (dim == spos)
            ptr_ = nullptr;
        return *this;
    }
    ConstIterator& ContiguousIncrementPos()
    {
        ptr_ += img_->elemsize_;
        return *this;
    }
};


template <typename T>
struct ContiguousIterator {
    uint8_t* ptr_;
    Image* img_;

    explicit ContiguousIterator(Image& img, std::vector<int> pos = {}) : img_{ &img }
    {
        if (!img_->contiguous_) {
            throw std::runtime_error("ContiguousIterator used on a non contiguous Image");
        }

        if (pos.empty()) { // Begin
            ptr_ = img.data_;
        }
        else {
            if (pos.size() != img_->dims_.size()) {
                throw std::runtime_error("ContiguousIterator starting pos has a wrong size");
            }
            if (pos == img_->dims_) { // End 
                ptr_ = img_->data_ + img_->datasize_;
            }
            else {
                ptr_ = img_->Ptr(pos);
            }
        }
    }
    ContiguousIterator& operator++() /* prefix */ { return ContiguousIncrementPos(); }
    T& operator* () const { return *reinterpret_cast<T*>(ptr_); }
    T* operator-> () const { return reinterpret_cast<T*>(ptr_); }
    bool operator==(const ContiguousIterator& rhs) const { return ptr_ == rhs.ptr_; }
    bool operator!=(const ContiguousIterator& rhs) const { return ptr_ != rhs.ptr_; }
private:
    ContiguousIterator& ContiguousIncrementPos()
    {
        ptr_ += img_->elemsize_;
        return *this;
    }
};

template <typename T>
struct ConstContiguousIterator {
    uint8_t* ptr_;
    const Image* img_;

    explicit ConstContiguousIterator(const Image& img, std::vector<int> pos = {}) : img_{ &img }
    {
        if (!img_->contiguous_) {
            throw std::runtime_error("ConstContiguousIterator used on a non contiguous Image");
        }

        if (pos.empty()) { // Begin
            ptr_ = img.data_;
        }
        else {
            if (pos.size() != img_->dims_.size()) {
                throw std::runtime_error("ConstContiguousIterator starting pos has a wrong size");
            }
            if (pos == img_->dims_) { // End 
                ptr_ = img_->data_ + img_->datasize_;
            }
            else {
                ptr_ = img_->Ptr(pos);
            }
        }
    }
    ConstContiguousIterator& operator++() /* prefix */ { return ContiguousIncrementPos(); }
    const T& operator* () const { return *reinterpret_cast<const T*>(ptr_); }
    const T* operator-> () const { return reinterpret_cast<const T*>(ptr_); }
    bool operator==(const ConstContiguousIterator& rhs) const { return ptr_ == rhs.ptr_; }
    bool operator!=(const ConstContiguousIterator& rhs) const { return ptr_ != rhs.ptr_; }
private:
    ConstContiguousIterator& ContiguousIncrementPos()
    {
        ptr_ += img_->elemsize_;
        return *this;
    }
};

} // namespace ecvl

#endif // !ECVL_ITERATORS_H_
