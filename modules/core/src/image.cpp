#include "ecvl/core/image.h"

namespace ecvl {

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
    Image tmp;
    // Check if rearranging is possible, else throw
    if (src.channels_ == "xyc" && channels == "cxy") {
        tmp = Image({ src.dims_[2], src.dims_[0], src.dims_[1] }, src.elemtype_, channels, src.colortype_);
        auto i = src.Begin<uint8_t>();
        auto plane_elems = src.dims_[0] * src.dims_[1];
        for (int ch = 0; ch < src.dims_[2]; ++ch) {
            auto ptr = tmp.data_ + ch;
            for (int el = 0; el < plane_elems; ++el) {
                memcpy(ptr, i.ptr_, tmp.elemsize_);
                ++i;
                ptr += tmp.strides_[1];
            }
        }
    }
    else if (src.channels_ == "cxy" && channels == "xyc")
    {
        tmp = Image({ src.dims_[1], src.dims_[2], src.dims_[0] }, src.elemtype_, channels, src.colortype_);
        auto i = src.Begin<uint8_t>();
        auto plane_elems = src.dims_[1] * src.dims_[2];
        for (int el = 0; el < plane_elems; ++el) {
            auto ptr = tmp.data_ + el;
            for (int ch = 0; ch < src.dims_[0]; ++ch) {
                memcpy(ptr, i.ptr_, tmp.elemsize_);
                ++i;
                ptr += tmp.strides_[2];
            }
        }
    }
    else {
        throw std::runtime_error("Not implemented");
    }

    dst = std::move(tmp);
}

Image& Image::Mul(double d, bool saturate)
{
    if (contiguous_) {
        switch (elemtype_)
        {
#define ECVL_TUPLE(name, ...) case DataType::name: return Image::Mul<ContiguousView<DataType::name>>(d, saturate);
#include "ecvl/core/datatype_existing_tuples.inc"
#undef ECVL_TUPLE
        default:
            throw std::runtime_error("How did you get here?");
        }
    }
    else {
        switch (elemtype_)
        {
#define ECVL_TUPLE(name, ...) case DataType::name: return Image::Mul<View<DataType::name>>(d, saturate);
#include "ecvl/core/datatype_existing_tuples.inc"
#undef ECVL_TUPLE
        default:
            throw std::runtime_error("How did you get here?");
        }
    }
}

} // namespace ecvl
