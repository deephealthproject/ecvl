#include <ecvl/eddl.h>

#include "ecvl/core/imgproc.h"
#include "ecvl/core/imgcodecs.h"
#include "ecvl/core/standard_errors.h"

#include <filesystem>
#include <iostream>

using namespace eddl;
using namespace std::filesystem;

namespace ecvl
{
void SetColorType(ColorType& c_type, const int& color_channels)
{
    if (c_type == ColorType::none) {
        switch (color_channels) {
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
    if (t->data->ndim != 3 && t->data->ndim != 4) {
        ECVL_ERROR_MSG "Tensor dims must be C x H x W or N x C x H x W";
    }

    Image img;

    switch (t->data->ndim) {
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
    if (t->data->ndim != 3 && t->data->ndim != 4) {
        ECVL_ERROR_MSG "Tensor dims must be C x H x W or N x C x H x W";
    }

    View<DataType::float32> v;

    switch (t->data->ndim) {
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
    if (img.dims_.size() != 3 && img.dims_.size() != 4) {
        ECVL_ERROR_MSG "Image must have 3 or 4 dimensions";
    }

    Image tmp;
    tensor t;
    CopyImage(img, tmp, DataType::float32);

    switch (tmp.dims_.size()) {
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

// Generic function to load a Dataset split into EDDL tensors
void DatasetToTensor(const Dataset& dataset, const std::vector<int>& size, const std::vector<int>& split, tensor& images, tensor& labels, ColorType ctype)
{
    if (size.size() != 2) {
        ECVL_ERROR_MSG "size must have 2 dimensions (height, width)";
    }
    tensor t;
    Image tmp;

    int n_samples = split.size();
    int n_channels = dataset.samples_[0].LoadImage(ctype).Channels();
    int n_classes = static_cast<int>(dataset.classes_.size());
    // Allocate memory for EDDL tensors
    images = T({ n_samples, n_channels, size[0], size[1] });
    labels = T({ n_samples, n_classes });

    // Fill tensors with data
    int i = 0;
    for (auto& index : split) {
        const Sample& elem = dataset.samples_[index];
        // Copy image into tensor (images)
        ResizeDim(elem.LoadImage(ctype), tmp, { size[1], size[0] });
        t = ImageToTensor(tmp);
        memcpy(images->data->ptr + t->data->size * i, t->data->ptr, t->data->size * sizeof(float));

        if (elem.label_) {
            // Copy labels into tensor (labels)
            vector<float> l(n_classes, 0);
            for (int j = 0; j < elem.label_.value().size(); ++j) {
                l[elem.label_.value()[j]] = 1;
            }
            memcpy(labels->data->ptr + l.size() * i, l.data(), l.size() * sizeof(float));
        }
        ++i;
    }
}

void TrainingToTensor(const Dataset& dataset, const std::vector<int>& size, tensor& stack, tensor& labels, ColorType ctype)
{
    DatasetToTensor(dataset, size, dataset.split_.training_, stack, labels, ctype);
}

void ValidationToTensor(const Dataset& dataset, const std::vector<int>& size, tensor& stack, tensor& labels, ColorType ctype)
{
    DatasetToTensor(dataset, size, dataset.split_.validation_, stack, labels, ctype);
}

void TestToTensor(const Dataset& dataset, const std::vector<int>& size, tensor& stack, tensor& labels, ColorType ctype)
{
    DatasetToTensor(dataset, size, dataset.split_.test_, stack, labels, ctype);
}

std::vector<int>& DLDataset::GetSplit()
{
    if (split_str_ == "training") {
        return this->split_.training_;
    }
    else if (split_str_ == "validation") {
        return this->split_.validation_;
    }
    else if (split_str_ == "test") {
        return this->split_.test_;
    }
    ECVL_ERROR_NOT_REACHABLE_CODE
}

void DLDataset::SetSplit(const string& split_str) {
    this->split_str_ = split_str;
}


void LoadBatch(DLDataset& dataset, const std::vector<int>& size, tensor& images, tensor& labels)
{
    if (size.size() != 2) {
        ECVL_ERROR_MSG "size must have 2 dimensions (height, width)";
    }
    Image tmp;
    int& bs = dataset.batch_size_;

    // Fill tensors with data
    int offset = 0;
    int start = dataset.current_batch_ * bs;

    for (int i = start; i < start + bs; ++i) {
        const int index = dataset.GetSplit()[i];
        const Sample& elem = dataset.samples_[index];
        // Copy image into tensor (images)
        ResizeDim(elem.LoadImage(dataset.ctype_), tmp, { size[1], size[0] });
        unique_ptr<LTensor> t(ImageToTensor(tmp));
        memcpy(images->data->ptr + t->data->size * offset, t->data->ptr, t->data->size * sizeof(float));

        if (elem.label_) {
            // Copy labels into tensor (labels)
            vector<float> l(dataset.classes_.size(), 0);
            for (int j = 0; j < elem.label_.value().size(); ++j) {
                l[elem.label_.value()[j]] = 1;
            }
            memcpy(labels->data->ptr + l.size() * offset, l.data(), l.size() * sizeof(float));
        }
        ++offset;
    }
}

} // namespace ecvl