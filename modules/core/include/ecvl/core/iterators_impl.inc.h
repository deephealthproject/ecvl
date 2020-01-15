/*
* ECVL - European Computer Vision Library
* Version: 0.1
* copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors: 
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/


template <typename T>
Iterator<T>::Iterator(Image& img, std::vector<int> pos) : img_{ &img }, pos_{ move(pos) }
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
template <typename T>
ConstIterator<T>::ConstIterator(const Image& img, std::vector<int> pos) : img_{ &img }, pos_{ move(pos) }
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
template <typename T>
ContiguousIterator<T>::ContiguousIterator(Image& img, std::vector<int> pos) : img_{ &img }
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
template <typename T>
ConstContiguousIterator<T>::ConstContiguousIterator(const Image& img, std::vector<int> pos) : img_{ &img }
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

template <typename T>
Iterator<T>& Iterator<T>::IncrementPos()
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
template <typename T>
ConstIterator<T>& ConstIterator<T>::IncrementPos()
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


template <typename T>
Iterator<T>& Iterator<T>::ContiguousIncrementPos()
{
    ptr_ += img_->elemsize_;
    return *this;
}
template <typename T>
ConstIterator<T>& ConstIterator<T>::ContiguousIncrementPos()
{
    ptr_ += img_->elemsize_;
    return *this;
}
template <typename T>
ContiguousIterator<T>& ContiguousIterator<T>::ContiguousIncrementPos()
{
    ptr_ += img_->elemsize_;
    return *this;
}
template <typename T>
ConstContiguousIterator<T>& ConstContiguousIterator<T>::ContiguousIncrementPos()
{
    ptr_ += img_->elemsize_;
    return *this;
}
