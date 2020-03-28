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

#include "ecvl/core/image.h"
#include "ecvl/core/datatype_matrix.h"

#include "ecvl/core/standard_errors.h"

namespace ecvl {
void Image::Create(const std::vector<int>& dims, DataType elemtype, std::string channels, ColorType colortype, const std::vector<float>& spacings)
{
    if (IsEmpty() || !IsOwner()) {
        *this = Image(dims, elemtype, std::move(channels), colortype, spacings);
        return;
    }
    else {
        if (!contiguous_) {
            *this = Image(dims, elemtype, std::move(channels), colortype, spacings);
            return;
        }
        else {
            // Compute datasize
            size_t new_datasize = DataTypeSize(elemtype);
            new_datasize = std::accumulate(begin(dims), end(dims), new_datasize, std::multiplies<size_t>());

            if (datasize_ != new_datasize) {
                datasize_ = new_datasize;
                mem_->Deallocate(data_);
                data_ = mem_->Allocate(new_datasize);
            }

            elemtype_ = elemtype;
            elemsize_ = DataTypeSize(elemtype_);
            dims_ = dims;   // A check could be added to save this copy
            spacings_ = spacings;
            channels_ = std::move(channels);
            colortype_ = colortype;
            datasize_ = new_datasize;

            // Compute strides
            strides_ = { elemsize_ };
            int dsize = vsize(dims_);
            for (int i = 0; i < dsize - 1; ++i) {
                strides_.push_back(strides_[i] * dims_[i]);
            }
            return;
        }
    }
}

void RearrangeAndCopy(const Image& src, Image& dst, const std::string& channels, DataType new_type)
{
    if (src.elemtype_ == DataType::none)
        throw std::runtime_error("Why should you copy a Image with none DataType into another?");

    if (&src == &dst) {
        if (src.elemtype_ != new_type && new_type != DataType::none) {
            throw std::runtime_error("src and dst cannot be the same image while changing the type");
            return;
        }
        if (src.channels_ == channels) {
            return;
        }
    }

    if (channels.size() != src.channels_.size()) {
        ECVL_ERROR_WRONG_PARAMS("channels.size() does not match src.channels_.size()")
    }

    if (src.channels_ == channels) {
        CopyImage(src, dst, new_type);
        return;
    }

    if (src.elemtype_ == new_type) {
        RearrangeChannels(src, dst, channels);
        return;
    }

    // bindings[new_pos] = old_pos
    std::vector<int> bindings(src.channels_.size());
    std::vector<int> new_dims(src.dims_);
    std::vector<float> new_spacings(src.spacings_.size());

    // Check if rearranging is required
    for (size_t old_pos = 0; old_pos < src.channels_.size(); old_pos++) {
        char c = src.channels_[old_pos];
        size_t new_pos = channels.find(c);
        if (new_pos == std::string::npos) {
            ECVL_ERROR_WRONG_PARAMS("channels contains wrong characters")
        }
        else {
            bindings[new_pos] = static_cast<int>(old_pos);
            new_dims[new_pos] = src.dims_[old_pos];
            if (new_spacings.size() == new_dims.size()) {   // spacings is not a mandatory field
                new_spacings[new_pos] = src.spacings_[old_pos];
            }
        }
    }

    if (new_type == DataType::none) {
        // Get type from dst or src
        if (dst.IsEmpty()) {
            RearrangeChannels(src, dst, channels);
            return;
        }
        if (src.dims_ != dst.dims_ || dst.channels_ != channels || src.elemtype_ != dst.elemtype_) {
            // Destination needs to be resized
            if (dst.mem_ == ShallowMemoryManager::GetInstance()) {
                throw std::runtime_error("Trying to resize an Image which doesn't own data.");
            }
            if (src.dims_ != dst.dims_ || dst.channels_ != channels || src.elemsize_ != dst.elemsize_) {
                dst = Image(new_dims, dst.elemtype_ == DataType::none ? src.elemtype_ : dst.elemtype_, channels, src.colortype_);
            }
        }
        if (src.colortype_ != dst.colortype_) {
            // Destination needs to change its color space
            if (dst.mem_ == ShallowMemoryManager::GetInstance()) {
                throw std::runtime_error("Trying to change color space on an Image which doesn't own data.");
            }
            dst.colortype_ = src.colortype_;
        }
    }
    else {
        if (dst.IsEmpty()) {
            dst = Image(new_dims, new_type, channels, src.colortype_);
        }
        else {
            if (src.dims_ != dst.dims_ || dst.channels_ != channels || dst.elemtype_ != new_type) {
                // Destination needs to be resized
                if (dst.mem_ == ShallowMemoryManager::GetInstance()) {
                    throw std::runtime_error("Trying to resize an Image which doesn't own data.");
                }
                if (src.dims_ != dst.dims_ || dst.channels_ != channels || dst.elemsize_ != DataTypeSize(new_type)) {
                    dst = Image(new_dims, new_type, channels, src.colortype_);
                }
                else {
                    dst.elemtype_ = new_type;
                }
            }
            if (src.colortype_ != dst.colortype_) {
                // Destination needs to change its color space
                if (dst.mem_ == ShallowMemoryManager::GetInstance()) {
                    throw std::runtime_error("Trying to change color space on an Image which doesn't own data.");
                }
                dst.colortype_ = src.colortype_;
            }
        }
    }

    static constexpr Table2D<StructRearrangeImage> table;
    table(src.elemtype_, dst.elemtype_)(src, dst, bindings);
}

void RearrangeChannels(const Image& src, Image& dst, const std::string& channels)
{
    // Check if rearranging is required
    if (src.channels_ == channels) {
        // if not, check if dst==src
        if (&src != &dst) { // if no, copy
            dst = src;
        }
        return;
    }

    if (channels.size() != src.channels_.size()) {
        ECVL_ERROR_WRONG_PARAMS("channels.size() does not match src.channels_.size()")
    }
    // bindings[new_pos] = old_pos
    std::vector<int> bindings(src.channels_.size());
    std::vector<int> new_dims(src.dims_.size());
    std::vector<float> new_spacings(src.spacings_.size());

    for (size_t old_pos = 0; old_pos < src.channels_.size(); old_pos++) {
        char c = src.channels_[old_pos];
        size_t new_pos = channels.find(c);
        if (new_pos == std::string::npos) {
            ECVL_ERROR_WRONG_PARAMS("channels contains wrong characters")
        }
        else {
            bindings[new_pos] = static_cast<int>(old_pos);
            new_dims[new_pos] = src.dims_[old_pos];
            if (new_spacings.size() == new_dims.size()) {   // spacings is not a mandatory field
                new_spacings[new_pos] = src.spacings_[old_pos];
            }
        }
    }

    Image tmp(new_dims, src.elemtype_, channels, src.colortype_, new_spacings);

    for (size_t tmp_pos = 0; tmp_pos < tmp.datasize_; tmp_pos += tmp.elemsize_) {
        int x = static_cast<int>(tmp_pos);
        int src_pos = 0;
        for (int i = vsize(tmp.dims_) - 1; i >= 0; i--) {
            src_pos += (x / tmp.strides_[i]) * src.strides_[bindings[i]];
            x %= tmp.strides_[i];
        }

        memcpy(tmp.data_ + tmp_pos, src.data_ + src_pos, tmp.elemsize_);
    }

    // TODO consider spacings
    // Check if rearranging is possible, else throw
    //if (src.channels_ == "xyc" && channels == "cxy") {
    //    std::vector<float> new_spacings;
    //    if (src.spacings_.size() == 3) {
    //        new_spacings = { src.spacings_[2], src.spacings_[0], src.spacings_[1] };
    //    }
    //    tmp = Image({ src.dims_[2], src.dims_[0], src.dims_[1] }, src.elemtype_, channels, src.colortype_, new_spacings);
    //    auto i = src.Begin<uint8_t>();
    //    auto plane_elems = src.dims_[0] * src.dims_[1];
    //    for (int ch = 0; ch < src.dims_[2]; ++ch) {
    //        auto ptr = tmp.data_ + ch * tmp.elemsize_;
    //        for (int el = 0; el < plane_elems; ++el) {
    //            memcpy(ptr, i.ptr_, tmp.elemsize_);
    //            ++i;
    //            ptr += tmp.strides_[1];
    //        }
    //    }
    //}
    //else if (src.channels_ == "cxy" && channels == "xyc")
    //{
    //    std::vector<float> new_spacings;
    //    if (src.spacings_.size() == 3) {
    //        new_spacings = { src.spacings_[1], src.spacings_[2], src.spacings_[0] };
    //    }
    //    tmp = Image({ src.dims_[1], src.dims_[2], src.dims_[0] }, src.elemtype_, channels, src.colortype_, new_spacings);
    //    auto i = src.Begin<uint8_t>();
    //    auto plane_elems = src.dims_[1] * src.dims_[2];
    //    for (int el = 0; el < plane_elems; ++el) {
    //        auto ptr = tmp.data_ + el * tmp.elemsize_;
    //        for (int ch = 0; ch < src.dims_[0]; ++ch) {
    //            memcpy(ptr, i.ptr_, tmp.elemsize_);
    //            ++i;
    //            ptr += tmp.strides_[2];
    //        }
    //    }
    //}
    //else {
    //    ECVL_ERROR_NOT_IMPLEMENTED
    //}

    dst = std::move(tmp);
}

void RearrangeChannels(const Image& src, Image& dst, const std::string& channels, DataType new_type)
{
    RearrangeAndCopy(src, dst, channels, new_type);
}

void CopyImage(const Image& src, Image& dst, DataType new_type)
{
    // TODO consider spacings

    if (src.elemtype_ == DataType::none)
        throw std::runtime_error("Why should you copy a Image with none DataType into another?");

    if (&src == &dst) {
        if (src.elemtype_ != new_type && new_type != DataType::none) {
            throw std::runtime_error("src and dst cannot be the same image while changing the type");
        }
        return;
    }

    if (new_type == DataType::none) {
        // Get type from dst or src
        if (dst.IsEmpty()) {
            dst = src;
            return;
        }
        if (src.dims_ != dst.dims_ || src.channels_ != dst.channels_ || src.elemtype_ != dst.elemtype_) {
            // Destination needs to be resized
            if (dst.mem_ == ShallowMemoryManager::GetInstance()) {
                throw std::runtime_error("Trying to resize an Image which doesn't own data.");
            }
            if (src.dims_ != dst.dims_ || src.channels_ != dst.channels_ || src.elemsize_ != dst.elemsize_) {
                dst = Image(src.dims_, dst.elemtype_ == DataType::none ? src.elemtype_ : dst.elemtype_, src.channels_, src.colortype_);
            }
        }
        if (src.colortype_ != dst.colortype_) {
            // Destination needs to change its color space
            if (dst.mem_ == ShallowMemoryManager::GetInstance()) {
                throw std::runtime_error("Trying to change color space on an Image which doesn't own data.");
            }
            dst.colortype_ = src.colortype_;
        }
    }
    else {
        if (dst.IsEmpty()) {
            dst = Image(src.dims_, new_type, src.channels_, src.colortype_);
        }
        else {
            if (src.dims_ != dst.dims_ || src.channels_ != dst.channels_ || dst.elemtype_ != new_type) {
                // Destination needs to be resized
                if (dst.mem_ == ShallowMemoryManager::GetInstance()) {
                    throw std::runtime_error("Trying to resize an Image which doesn't own data.");
                }
                if (src.dims_ != dst.dims_ || src.channels_ != dst.channels_ || dst.elemsize_ != DataTypeSize(new_type)) {
                    dst = Image(src.dims_, new_type, src.channels_, src.colortype_);
                }
                else {
                    dst.elemtype_ = new_type;
                }
            }
            if (src.colortype_ != dst.colortype_) {
                // Destination needs to change its color space
                if (dst.mem_ == ShallowMemoryManager::GetInstance()) {
                    throw std::runtime_error("Trying to change color space on an Image which doesn't own data.");
                }
                dst.colortype_ = src.colortype_;
            }
        }
    }

    static constexpr Table2D<StructCopyImage> table;
    table(src.elemtype_, dst.elemtype_)(src, dst);
}

void CopyImage(const Image& src, Image& dst, DataType new_type, const std::string& channels)
{
    RearrangeAndCopy(src, dst, channels, new_type);
}

Image& Image::operator+=(const Image& rhs)
{
    Add(rhs);
    return *this;
}

Image& Image::operator-=(const Image& rhs)
{
    Sub(rhs);
    return *this;
}

Image& Image::operator*=(const Image& rhs)
{
    Mul(rhs);
    return *this;
}

Image& Image::operator/=(const Image& rhs)
{
    Div(rhs);
    return *this;
}

Image operator+(Image lhs, const Image& rhs)
{
    lhs += rhs;
    return lhs;
}

Image operator-(Image lhs, const Image& rhs)
{
    lhs -= rhs;
    return lhs;
}

Image operator*(Image lhs, const Image& rhs)
{
    lhs *= rhs;
    return lhs;
}

Image operator/(Image lhs, const Image& rhs)
{
    lhs /= rhs;
    return lhs;
}
} // namespace ecvl