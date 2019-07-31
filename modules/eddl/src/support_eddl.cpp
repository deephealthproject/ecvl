#include <ecvl/eddl.h>

#include "ecvl/core/imgproc.h"
#include "ecvl/core/imgcodecs.h"
#include "ecvl/core/standard_errors.h"

#include <iostream>

using namespace eddl;

namespace ecvl
{
    void SetColorType(ColorType& c_type, const int &color_channels)
    {
        if (c_type == ColorType::none)
        {
            switch (color_channels)
            {
            case 1:     c_type = ColorType::GRAY;      break;
            case 3:     c_type = ColorType::BGR;       break;
            case 4:     c_type = ColorType::RGBA;      break;
            default:
                c_type = ColorType::none;
            }
        }
    }

    Image TensorToImage(tensor& t, ColorType c_type)
    {
        if (t->data->ndim != 3 && t->data->ndim != 4)
        {
            ECVL_ERROR_MSG "Tensor dims must be C x H x W or N x C x H x W";
        }

        Image img;

        switch (t->data->ndim)
        {
        case 3:
            if (c_type == ColorType::none)
                SetColorType(c_type, t->data->shape[0]);
            img.Create({ t->data->shape[2], t->data->shape[1], t->data->shape[0] }, DataType::float32, "xyc", c_type);
            break;
        case 4:
            if (c_type == ColorType::none)
                SetColorType(c_type, t->data->shape[1]);
            img.Create({ t->data->shape[3], t->data->shape[2], t->data->shape[0], t->data->shape[1] }, DataType::float32, "xyzc", c_type);
            break;
        default:
            ECVL_ERROR_NOT_REACHABLE_CODE
        }

        memcpy(img.data_, t->data->ptr, img.datasize_);

        return img;
    }

    View<DataType::float32> TensorToView(tensor& t, ColorType c_type)
    {
        if (t->data->ndim != 3 && t->data->ndim != 4)
        {
            ECVL_ERROR_MSG "Tensor dims must be C x H x W or N x C x H x W";
        }

        View<DataType::float32> v;

        switch (t->data->ndim)
        {
        case 3:
            if (c_type == ColorType::none)
                SetColorType(c_type, t->data->shape[0]);
            v.dims_.push_back(t->data->shape[2]);
            v.dims_.push_back(t->data->shape[1]);
            v.dims_.push_back(t->data->shape[0]);
            v.channels_ = "xyc";
            break;
        case 4:
            if (c_type == ColorType::none)
                SetColorType(c_type, t->data->shape[1]);
            v.dims_.push_back(t->data->shape[3]);
            v.dims_.push_back(t->data->shape[2]);
            v.dims_.push_back(t->data->shape[0]);
            v.dims_.push_back(t->data->shape[1]);
            v.channels_ = "xyzc";
            break;
        default:
            ECVL_ERROR_NOT_REACHABLE_CODE
        }

        v.colortype_ = c_type;
        v.data_ = (uint8_t*)t->data->ptr;
        v.elemtype_ = DataType::float32;
        v.elemsize_ = DataTypeSize(DataType::float32);
        v.spacings_ = {};
        v.contiguous_ = true;
        v.meta_ = { nullptr };
        v.mem_ = ShallowMemoryManager::GetInstance();

        // Compute strides
        v.strides_ = { v.elemsize_ };
        int dsize = v.dims_.size();
        for (int i = 0; i < dsize - 1; ++i) {
            v.strides_.push_back(v.strides_[i] * v.dims_[i]);
        }

        // Compute datasize
        v.datasize_ = v.elemsize_;
        v.datasize_ = std::accumulate(begin(v.dims_), end(v.dims_), v.datasize_, std::multiplies<size_t>());

        return v;
    }

    tensor ImageToTensor(const Image& img)
    {
        if (img.dims_.size() != 3 && img.dims_.size() != 4)
        {
            ECVL_ERROR_MSG "Image must have 3 or 4 dimensions";
        }

        Image tmp; tensor t;
        CopyImage(img, tmp, DataType::float32);

        switch (tmp.dims_.size())
        {
        case 3:
            if (tmp.channels_ != "xyc")
                RearrangeChannels(tmp, tmp, "xyc");
            t = T({ tmp.dims_[2], tmp.dims_[1], tmp.dims_[0] });
            break;
        case 4:
            if (tmp.channels_ != "xyzc")
                RearrangeChannels(tmp, tmp, "xyzc");
            t = T({ tmp.dims_[2], tmp.dims_[3], tmp.dims_[1], tmp.dims_[0] });
            break;
        default:
            ECVL_ERROR_NOT_REACHABLE_CODE
        }

        memcpy(t->data->ptr, tmp.data_, tmp.datasize_);
        return t;
    }

    tensor DatasetToTensor(vector<string> dataset, const std::vector<int>& dims)
    {
        if (dims.size() != 4)
            ECVL_ERROR_MSG "dims must have 4 dimensions (n_samples, color_channels, height, width )";

        Image tmp;
        tensor t; 
        int i = 0;

        tensor stack = T({ dims });

        for (auto& elem : dataset) {
            ImRead(elem, tmp);
            ResizeDim(tmp, tmp, { dims[3], dims[2] });
            t = ImageToTensor(tmp);
            memcpy(stack->data->ptr + t->data->size * i, t->data->ptr, t->data->size * sizeof(float));
            ++i;
        }

        return stack;
    }
}