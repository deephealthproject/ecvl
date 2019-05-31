#include "ecvl/core/image.h"
#include "ecvl/core/datatype_matrix.h"


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

void CopyImage(Image& src, Image& dst, DataType new_type)
{
    if (&src == &dst)
        throw std::runtime_error("src and dst cannot be the same image");

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
                dst = Image(src.dims_, src.elemtype_, src.channels_, src.colortype_);
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

} // namespace ecvl
