#include "ecvl/core/image.h"
#include "ecvl/core/datatype_matrix.h"

#include "ecvl/core/standard_errors.h"

namespace ecvl {

void Image::Create(const std::vector<int>& dims, DataType elemtype, std::string channels, ColorType colortype) {

    if (mem_ == ShallowMemoryManager::GetInstance()) {
        *this = Image(dims, elemtype, std::move(channels), colortype);
        return;
    }
    else {

        if (!contiguous_) {
            *this = Image(dims, elemtype, std::move(channels), colortype);
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
            channels_ = std::move(channels);
            colortype_ = colortype;
            datasize_ = new_datasize;

            // Compute strides
            strides_ = { elemsize_ };
            int dsize = dims_.size();
            for (int i = 0; i < dsize - 1; ++i) {
                strides_.push_back(strides_[i] * dims_[i]);
            }
            return;
        }
    }
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
        throw std::runtime_error(ERROR_NOT_IMPLEMENTED);
    }

    dst = std::move(tmp);
}

void CopyImage(const Image& src, Image& dst, DataType new_type)
{
    if (src.elemtype_ == DataType::none)
        throw std::runtime_error("Why should you copy a Image with none DataType into another?");

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

} // namespace ecvl
