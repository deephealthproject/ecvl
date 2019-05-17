
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
Iterator<T>& Iterator<T>::ContiguousIncrementPos()
{
    ptr_ += img_->elemsize_;
    return *this;
}

