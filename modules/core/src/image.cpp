/*
* ECVL - European Computer Vision Library
* Version: 0.2.1
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
#include <iostream>
#include "ecvl/core/datatype_matrix.h"

#include "ecvl/core/standard_errors.h"

namespace ecvl
{
	using namespace std;
void Image::Create(const std::vector<int>& dims, DataType elemtype, std::string channels, ColorType colortype,
    const std::vector<float>& spacings, Device dev)
{
    if (IsEmpty() || !IsOwner() || dev_ != dev || !contiguous_) {
        *this = Image(dims, elemtype, std::move(channels), colortype, spacings, dev);
    }
    else {
        elemtype_ = elemtype;
        elemsize_ = DataTypeSize(elemtype_);
        dims_ = dims;   // A check could be added to save this copy
        spacings_ = spacings;
        channels_ = std::move(channels);
        colortype_ = colortype;

        // Compute new datasize
        size_t new_datasize = GetDefaultDatasize();

        if (datasize_ != new_datasize) {
			cout << "create imgen con nuevo datasize" << endl;
            datasize_ = new_datasize;
            hal_->MemDeallocate(data_);
            data_ = hal_->MemAllocate(new_datasize);
        }

        datasize_ = new_datasize;

        SetDefaultStrides();
    }
}

void RearrangeBindings(const Image& src, const std::string& channels, std::vector<int>& bindings, std::vector<int>& new_dims, std::vector<float>& new_spacings)
{
    if (channels.size() != src.channels_.size()) {
        ECVL_ERROR_WRONG_PARAMS("channels.size() does not match src.channels_.size()")
    }

    if (src.IsEmpty()) {
        ECVL_ERROR_EMPTY_IMAGE
    }

    // bindings[new_pos] = old_pos
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
}

void CopyImageCreateDst(const Image& src, Image& dst, DataType new_type, const std::vector<int>& dims = {}, const std::string& channels = "")
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

    std::vector<int> new_dims = dims.empty() ? src.dims_ : dims;
    std::string new_channels = channels.empty() ? src.channels_ : channels;

    if (new_type == DataType::none) {
        // Get type from dst or src
        if (dst.IsEmpty()) {
            if (channels.empty()) {
                dst = src;
                return;
            }
        }
        if (src.dims_ != dst.dims_ || new_channels != dst.channels_ || src.elemtype_ != dst.elemtype_) {
            // Destination needs to be resized
            if (!dst.IsOwner()) {
                throw std::runtime_error("Trying to resize an Image which doesn't own data.");
            }
            if (src.dims_ != dst.dims_ || new_channels != dst.channels_ || src.elemsize_ != dst.elemsize_) {
                dst = Image(new_dims, dst.elemtype_ == DataType::none ? src.elemtype_ : dst.elemtype_,
                    new_channels, src.colortype_, src.spacings_, src.dev_);
            }
        }
        if (src.colortype_ != dst.colortype_) {
            // Destination needs to change its color space
            if (!dst.IsOwner()) {
                throw std::runtime_error("Trying to change color space on an Image which doesn't own data.");
            }
            dst.colortype_ = src.colortype_;
        }
    }
    else {
        if (dst.IsEmpty()) {
            dst = Image(new_dims, new_type, new_channels, src.colortype_, src.spacings_, src.dev_);
        }
        else {
            if (src.dims_ != dst.dims_ || new_channels != dst.channels_ || dst.elemtype_ != new_type) {
                // Destination needs to be resized
                if (!dst.IsOwner()) {
                    throw std::runtime_error("Trying to resize an Image which doesn't own data.");
                }
                if (src.dims_ != dst.dims_ || new_channels != dst.channels_ || dst.elemsize_ != DataTypeSize(new_type)) {
                    dst = Image(new_dims, new_type, new_channels, src.colortype_, src.spacings_, src.dev_);
                }
                else {
                    dst.elemtype_ = new_type;
                }
            }
            if (src.colortype_ != dst.colortype_) {
                // Destination needs to change its color space
                if (!dst.IsOwner()) {
                    throw std::runtime_error("Trying to change color space on an Image which doesn't own data.");
                }
                dst.colortype_ = src.colortype_;
            }
        }
    }
}

void RearrangeAndCopy(const Image& src, Image& dst, const std::string& channels, DataType new_type)
{
    // Check if rearranging is required
    if (src.channels_ == channels) {
        CopyImage(src, dst, new_type);
        return;
    }

    // Check if copy is reqiured
    if (src.elemtype_ == new_type) {
        RearrangeChannels(src, dst, channels);
        return;
    }

    std::vector<int> bindings(src.channels_.size());
    std::vector<int> new_dims(src.dims_.size());
    std::vector<float> new_spacings(src.spacings_.size());

    RearrangeBindings(src, channels, bindings, new_dims, new_spacings);
    CopyImageCreateDst(src, dst, new_type, new_dims, channels);

    dst.hal_->RearrangeChannels(src, dst, bindings);
}

void RearrangeChannels(const Image& src, Image& dst, const std::string& channels)
{
    // Check if rearranging is required
    if (src.channels_ == channels) {
        // No rearranging required, it's just a copy
        dst = src;
        return;
    }

    if (src.dev_ != dst.dev_ && dst.dev_ != Device::NONE) {
        ECVL_ERROR_DIFFERENT_DEVICES
    }

    std::vector<int> bindings(src.channels_.size());
    std::vector<int> new_dims(src.dims_.size());
    std::vector<float> new_spacings(src.spacings_.size());

    RearrangeBindings(src, channels, bindings, new_dims, new_spacings);

    Image tmp(new_dims, src.elemtype_, channels, src.colortype_, new_spacings, src.dev_);

    src.hal_->RearrangeChannels(src, tmp, bindings);

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
    CopyImageCreateDst(src, dst, new_type);
    dst.hal_->CopyImage(src, dst);
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

Image Image::operator-() const
{
    Image ret(*this);
    ret.Neg();
    return ret;
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