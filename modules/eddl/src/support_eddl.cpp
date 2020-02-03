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
void TensorToImage(tensor& t, Image& img)
{
    switch (t->ndim) {
    case 3:
        img.Create({ t->shape[2], t->shape[1], t->shape[0] }, DataType::float32, "xyo", ColorType::none);
        break;
    case 4:
        img.Create({ t->shape[3], t->shape[2], t->shape[0] * t->shape[1] }, DataType::float32, "xyo", ColorType::none);
        break;
    default:
        ECVL_ERROR_MSG "Tensor dims must be C x H x W or N x C x H x W";
    }

    memcpy(img.data_, t->ptr, img.datasize_);
}

void TensorToView(tensor& t, View<DataType::float32>& v)
{
    switch (t->ndim) {
    case 3:
        v.Create({ t->shape[2], t->shape[1], t->shape[0] }, "xyo", ColorType::none, (uint8_t*)t->ptr);
        break;
    case 4:
        v.Create({ t->shape[3], t->shape[2], t->shape[0] * t->shape[1] }, "xyo", ColorType::none, (uint8_t*)t->ptr);
        break;
    default:
        ECVL_ERROR_MSG "Tensor dims must be C x H x W or N x C x H x W";
    }
}

void ImageToTensor(const Image& img, tensor& t)
{
    Image tmp;
    string channels;

    if (img.dims_.size() != 3) {
        ECVL_ERROR_MSG "Image must have 3 dimensions 'xy[czo]' (in any order)";
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    // If img is one of: cxy, cyx, xcy, ycx... convert it to xyc
    if (img.channels_.find('c') != string::npos && img.channels_ != "xyc") {
        channels = "xyc";
    }
    // If img is one of: zxy, zyx, xzy, yzx... convert it to xyz
    else if (img.channels_.find('z') != string::npos && img.channels_ != "xyz") {
        channels = "xyz";
    }
    // If img is one of: oxy, oyx, xoy, yox... convert it to xyo
    else if (img.channels_.find('o') != string::npos && img.channels_ != "xyo") {
        channels = "xyo";
    }
    else if (img.channels_.find('o') == string::npos &&
        img.channels_.find('c') == string::npos &&
        img.channels_.find('z') == string::npos) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if (channels.size() > 0) {
        RearrangeChannels(img, tmp, channels, DataType::float32);
    }
    else {
        CopyImage(img, tmp, DataType::float32);
    }
    t = eddlT::create({ tmp.dims_[2], tmp.dims_[1], tmp.dims_[0] });

    memcpy(t->ptr, tmp.data_, tmp.datasize_);
}

void ImageToTensor(const Image& img, tensor& t, const int& offset)
{
    Image tmp;
    int tot_dims = 0;
    string channels;

    if (img.dims_.size() != 3) {
        ECVL_ERROR_MSG "Image must have 3 dimensions 'xy[czo]' (in any order)";
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    // If img is one of: cxy, cyx, xcy, ycx... convert it to xyc
    if (img.channels_.find('c') != string::npos && img.channels_ != "xyc") {
        channels = "xyc";
    }
    // If img is one of: zxy, zyx, xzy, yzx... convert it to xyz
    else if (img.channels_.find('z') != string::npos && img.channels_ != "xyz") {
        channels = "xyz";
    }
    // If img is one of: oxy, oyx, xoy, yox... convert it to xyo
    else if (img.channels_.find('o') != string::npos && img.channels_ != "xyo") {
        channels = "xyo";
    }
    else if (img.channels_.find('o') == string::npos &&
        img.channels_.find('c') == string::npos &&
        img.channels_.find('z') == string::npos) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if (channels.size() > 0) {
        RearrangeChannels(img, tmp, channels, DataType::float32);
    }
    else {
        CopyImage(img, tmp, DataType::float32);
    }

    tot_dims = accumulate(img.dims_.begin(), img.dims_.end(), 1, std::multiplies<int>());

    for (int i = 0; i < t->ndim; ++i) {
        if (t->shape[i] != img.dims_[i]) {
            // ERROR
        }
    }

    memcpy(t->ptr + tot_dims * offset, tmp.data_, tot_dims * sizeof(float));
}

/** @cond HIDDEN_SECTIONS */
// Generic function to load a Dataset split into EDDL tensors
void DatasetToTensor(const Dataset& dataset, const std::vector<int>& size, const std::vector<int>& split, tensor& images, tensor& labels, ColorType ctype)
{
    if (size.size() != 2) {
        ECVL_ERROR_MSG "size must have 2 dimensions (height, width)";
    }
    Image tmp;

    int n_samples = split.size();
    int n_channels = dataset.samples_[0].LoadImage(ctype, false).Channels();
    int n_classes = static_cast<int>(dataset.classes_.size());
    // Allocate memory for EDDL tensors
    images = eddlT::create({ n_samples, n_channels, size[0], size[1] });
    labels = eddlT::create({ n_samples, n_classes });

    // Fill tensors with data
    int i = 0;
    for (auto& index : split) {
        const Sample& elem = dataset.samples_[index];
        // Copy image into tensor (images)
        ResizeDim(elem.LoadImage(ctype, false), tmp, { size[1], size[0] });
        ImageToTensor(tmp, images, i);
        if (elem.label_) {
            // Copy labels into tensor (labels)
            vector<float> l(n_classes, 0);
            for (int j = 0; j < elem.label_.value().size(); ++j) {
                l[elem.label_.value()[j]] = 1;
            }
            memcpy(labels->ptr + l.size() * i, l.data(), l.size() * sizeof(float));
        }
        ++i;
    }
}
/** @endcond */

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
    if (current_split_ == 0) {
        return this->split_.training_;
    }
    else if (current_split_ == 1) {
        return this->split_.validation_;
    }
    else {
        return this->split_.test_;
    }
}

void DLDataset::SetSplit(const string& split_str)
{
    if (split_str == "training") {
        this->current_split_ = 0;
    }
    else if (split_str == "validation") {
        this->current_split_ = 1;
    }
    else if (split_str == "test") {
        this->current_split_ = 2;
    }
}

void DLDataset::ResetCurrentBatch()
{
    this->current_batch_[current_split_] = 0;
}

void DLDataset::ResetAllBatches()
{
    this->current_batch_.fill(0);
}

void DLDataset::LoadBatch(tensor& images, tensor& labels)
{
    if (resize_dims_.size() != 2) {
        ECVL_ERROR_MSG "resize_dims_ must have 2 dimensions (height, width)";
    }

    int& bs = batch_size_;
    Image tmp;
    int offset = 0, start = 0;

    // Move to next samples
    start = current_batch_[current_split_] * bs;
    current_batch_[current_split_]++;

    // Fill tensors with data
    for (int i = start; i < start + bs; ++i) {
        const int index = GetSplit()[i];
        const Sample& elem = samples_[index];
        // Read and resize (HxW -> WxH) image
        ResizeDim(elem.LoadImage(ctype_, false), tmp, { resize_dims_[1], resize_dims_[0] });
        // Copy image into tensor (images)
        ImageToTensor(tmp, images, offset);

        if (elem.label_) {
            // Copy labels into tensor (labels)
            vector<float> lab(classes_.size(), 0);
            for (int j = 0; j < elem.label_.value().size(); ++j) {
                lab[elem.label_.value()[j]] = 1;
            }
            memcpy(labels->ptr + lab.size() * offset, lab.data(), lab.size() * sizeof(float));
        }
        else if (elem.label_path_) {
            // Read and resize (HxW -> WxH) ground truth image
            ResizeDim(elem.LoadImage(ctype_gt_, true), tmp, { resize_dims_[1], resize_dims_[0] });
            // Copy labels into tensor (labels)
            ImageToTensor(tmp, labels, offset);
        }
        ++offset;
    }
}
} // namespace ecvl