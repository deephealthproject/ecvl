/*
* ECVL - European Computer Vision Library
* Version: 0.1
* copyright (c) 2020, Universit� degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include <ecvl/support_eddl.h>

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
        cerr << ECVL_ERROR_MSG "Tensor dims must be C x H x W or N x C x H x W" << endl;
        ECVL_ERROR_INCOMPATIBLE_DIMENSIONS
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
        cerr << ECVL_ERROR_MSG "Tensor dims must be C x H x W or N x C x H x W" << endl;
        ECVL_ERROR_INCOMPATIBLE_DIMENSIONS
    }
}

void ImageToTensor(const Image& img, tensor& t)
{
    Image tmp;
    string channels;

    if (img.dims_.size() != 3) {
        cerr << ECVL_ERROR_MSG "Image must have 3 dimensions 'xy[czo]' (in any order)" << endl;
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
        cerr << ECVL_ERROR_MSG "Image must have 3 dimensions 'xy[czo]' (in any order)" << endl;
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

    // Check if the current image exceeds the total size of the tensor
    if (t->size < tot_dims * (offset + 1)) {
        cerr << ECVL_ERROR_MSG "Size of the images exceeds those of the tensor" << endl;
        ECVL_ERROR_INCOMPATIBLE_DIMENSIONS
    }

    memcpy(t->ptr + tot_dims * offset, tmp.data_, tot_dims * sizeof(float));
}

std::vector<int>& DLDataset::GetSplit()
{
    if (current_split_ == SplitType::training) {
        return this->split_.training_;
    }
    else if (current_split_ == SplitType::validation) {
        return this->split_.validation_;
    }
    else {
        return this->split_.test_;
    }
}

void DLDataset::SetSplit(const SplitType& split)
{
    this->current_split_ = split;
}

void DLDataset::ResetCurrentBatch()
{
    this->current_batch_[+current_split_] = 0;
}

void DLDataset::ResetAllBatches()
{
    this->current_batch_.fill(0);
}

void DLDataset::LoadBatch(tensor& images, tensor& labels)
{
    if (resize_dims_.size() != 2) {
        cerr << ECVL_ERROR_MSG "resize_dims_ must have 2 dimensions (height, width)" << endl;
        ECVL_ERROR_INCOMPATIBLE_DIMENSIONS
    }

    int& bs = batch_size_;
    Image img, gt;
    int offset = 0, start = 0;

    // Check if tensors size matches with batch dimensions
    // size of images tensor must be equal to batch_size * number_of_image_channels * image_width * image_height
    if (images->size != bs * n_channels_ * resize_dims_[0] * resize_dims_[1]) {
        cerr << ECVL_ERROR_MSG "images tensor must have N = batch_size, C = number_of_image_channels, H = image_height, W = image_width" << endl;
        ECVL_ERROR_INCOMPATIBLE_DIMENSIONS
    }

    // if it is a classification problem, size of labels tensor must be equal to batch_size * number_of_classes
    if (samples_[0].label_.has_value()) {
        if (labels->size != bs * classes_.size()) {
            cerr << ECVL_ERROR_MSG "labels tensor must have N = batch_size, C = number_of_classes" << endl;
            ECVL_ERROR_INCOMPATIBLE_DIMENSIONS
        }
    }
    // otherwise is a segmentation problem so size of labels tensor must be equal to batch_size * number_of_label_channels * image_width * image_height
    else if (labels->size != bs * n_channels_gt_ * resize_dims_[0] * resize_dims_[1]) {
        cerr << ECVL_ERROR_MSG "labels tensor must have N = batch_size, C = number_of_label_channels, H = image_height, W = image_width" << endl;
        ECVL_ERROR_INCOMPATIBLE_DIMENSIONS
    }

    // Move to next samples
    start = current_batch_[+current_split_] * bs;
    ++current_batch_[+current_split_];

    if (GetSplit().size() < start + bs) {
        cerr << ECVL_ERROR_MSG "Batch size is not even with the number of samples. Hint: loop through `num_batches = num_samples / batch_size;`" << endl;
        ECVL_ERROR_CANNOT_LOAD_IMAGE
    }

    // Fill tensors with data
    for (int i = start; i < start + bs; ++i) {
        const int index = GetSplit()[i];
        const Sample& elem = samples_[index];
        // Read and resize (HxW -> WxH) image
        img = elem.LoadImage(ctype_, false);

        if (elem.label_) {
            // Apply chain of augmentations only to sample image
            augs_.Apply(current_split_, img);

            // Copy labels into tensor (labels)
            vector<float> lab(classes_.size(), 0);
            for (int j = 0; j < elem.label_.value().size(); ++j) {
                lab[elem.label_.value()[j]] = 1;
            }
            memcpy(labels->ptr + lab.size() * offset, lab.data(), lab.size() * sizeof(float));
        }
        else if (elem.label_path_) {
            gt = elem.LoadImage(ctype_gt_, true);

            // Apply chain of augmentations only to sample image
            augs_.Apply(current_split_, img, gt);

            // Copy labels into tensor (labels)
            ImageToTensor(gt, labels, offset);
        }

        // Copy image into tensor (images)
        ImageToTensor(img, images, offset);

        ++offset;
    }
}
} // namespace ecvl