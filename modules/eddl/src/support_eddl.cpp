/*
* ECVL - European Computer Vision Library
* Version: 1.0.3
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include <ecvl/support_eddl.h>

#include "ecvl/core/filesystem.h"
#include "ecvl/core/imgcodecs.h"
#include "ecvl/core/imgproc.h"
#include "ecvl/core/standard_errors.h"

#include <algorithm>
#include <iostream>

using namespace eddl;
using namespace ecvl::filesystem;
using namespace std;

namespace ecvl
{
#define ECVL_ERROR_START_ALREADY_ACTIVE throw std::runtime_error(ECVL_ERROR_MSG "Trying to start the producer threads when they are already running!");
#define ECVL_ERROR_STOP_ALREADY_END throw std::runtime_error(ECVL_ERROR_MSG "Trying to stop the producer threads when they are already ended!");
default_random_engine DLDataset::re_(random_device{}());

void TensorToImage(const Tensor* t, Image& img)
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

void TensorToView(const Tensor* t, View<DataType::float32>& v)
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

void ImageToTensor(const Image& img, Tensor*& t)
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
    t = new Tensor({ tmp.dims_[2], tmp.dims_[1], tmp.dims_[0] });

    memcpy(t->ptr, tmp.data_, tmp.datasize_);
}

void ImageToTensor(const Image& img, Tensor*& t, const int& offset)
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
    if (t->size < static_cast<unsigned>(tot_dims * (offset + 1))) {
        cerr << ECVL_ERROR_MSG "Size of the images exceeds those of the tensor" << endl;
        ECVL_ERROR_INCOMPATIBLE_DIMENSIONS
    }

    memcpy(t->ptr + tot_dims * offset, tmp.data_, tot_dims * sizeof(float));
}

void DLDataset::ResetBatch(const any& split, bool shuffle)
{
    int index = GetSplitIndex(split);
    this->current_batch_.at(index) = 0;

    if (shuffle) {
        std::shuffle(begin(GetSplit(index)), end(GetSplit(index)), re_);
    }

    for (auto& tc : splits_tc_[index]) {
        tc.Reset();
    }
}

void DLDataset::ResetAllBatches(bool shuffle)
{
    fill(current_batch_.begin(), current_batch_.end(), 0);

    for (int split_index = 0; split_index < vsize(split_); ++split_index) {
        if (shuffle) {
            std::shuffle(begin(GetSplit(split_index)), end(GetSplit(split_index)), re_);
        }
        for (auto& tc : splits_tc_[split_index]) {
            tc.Reset();
        }
    }
}

void DLDataset::LoadBatch(Tensor*& images, Tensor*& labels)
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
    if (samples_[0].label_ != nullopt) {
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

    start = current_batch_[current_split_] * bs;
    ++current_batch_[current_split_];

    if (vsize(GetSplit()) < start + bs) {
        cerr << ECVL_ERROR_MSG "Batch size is not even with the number of samples. Hint: loop through `num_batches = num_samples / batch_size;`" << endl;
        ECVL_ERROR_CANNOT_LOAD_IMAGE
    }

    // Fill tensors with data
    for (int i = start; i < start + bs; ++i) {
        // Read the image
        const int index = GetSplit()[i];
        Sample& elem = samples_[index];
        img = elem.LoadImage(ctype_, false);

        // Classification problem
        if (elem.label_) {
            // Apply chain of augmentations only to sample image
            augs_.Apply(current_split_, img);

            // Copy label into tensor (labels)
            vector<float> lab(classes_.size(), 0);
            for (int j = 0; j < vsize(elem.label_.value()); ++j) {
                lab[elem.label_.value()[j]] = 1;
            }
            memcpy(labels->ptr + lab.size() * offset, lab.data(), lab.size() * sizeof(float));
        }
        // Segmentation problem
        else if (elem.label_path_) {
            // Read the ground truth
            gt = elem.LoadImage(ctype_gt_, true);

            // Apply chain of augmentations to sample image and corresponding ground truth
            augs_.Apply(current_split_, img, gt);

            // Copy label into tensor (labels)
            ImageToTensor(gt, labels, offset);
        }

        // Copy image into tensor (images)
        ImageToTensor(img, images, offset);

        ++offset;
    }
}

void DLDataset::LoadBatch(Tensor*& images)
{
    if (resize_dims_.size() != 2) {
        cerr << ECVL_ERROR_MSG "resize_dims_ must have 2 dimensions (height, width)" << endl;
        ECVL_ERROR_INCOMPATIBLE_DIMENSIONS
    }

    int& bs = batch_size_;
    Image img;
    int offset = 0, start = 0;

    // Check if tensors size matches with batch dimensions
    // size of images tensor must be equal to batch_size * number_of_image_channels * image_width * image_height
    if (images->size != bs * n_channels_ * resize_dims_[0] * resize_dims_[1]) {
        cerr << ECVL_ERROR_MSG "images tensor must have N = batch_size, C = number_of_image_channels, H = image_height, W = image_width" << endl;
        ECVL_ERROR_INCOMPATIBLE_DIMENSIONS
    }

    // Move to next samples

    start = current_batch_[current_split_] * bs;
    ++current_batch_[current_split_];

    if (vsize(GetSplit()) < start + bs) {
        cerr << ECVL_ERROR_MSG "Batch size is not even with the number of samples. Hint: loop through `num_batches = num_samples / batch_size;`" << endl;
        ECVL_ERROR_CANNOT_LOAD_IMAGE
    }

    // Fill tensors with data
    for (int i = start; i < start + bs; ++i) {
        // Read the image
        const int index = GetSplit()[i];
        Sample& elem = samples_[index];
        img = elem.LoadImage(ctype_, false);

        // Apply chain of augmentations only to sample image
        augs_.Apply(current_split_, img);

        // Copy image into tensor (images)
        ImageToTensor(img, images, offset);

        ++offset;
    }
}

Image MakeGrid(Tensor*& t, int cols, bool normalize)
{
    const auto batch_size = t->shape[0];
    cols = std::min(batch_size, cols);
    const auto rows = static_cast<int>(std::ceil(static_cast<double>(batch_size) / cols));

    Image image_t;
    vector<Image> vimages;
    for (int r = 0, b = 0; r < rows; ++r) {
        vector<Image> himages;
        for (int c = 0; c < cols; ++c) {
            Tensor* tensor_t;
            if (b < batch_size) {
                tensor_t = t->select({ to_string(b) });
                TensorToImage(tensor_t, image_t);
                if (normalize) {
                    ScaleTo(image_t, image_t, 0, 1);
                }
                image_t.Mul(255.);
                image_t.channels_ = "xyc";
                image_t.ConvertTo(DataType::uint8);
                delete tensor_t;
            }
            else {
                image_t = Image({ t->shape[3],t->shape[2],t->shape[1] }, DataType::uint8, "xyc", ColorType::none);
                image_t.SetTo(0);
            }
            himages.push_back(image_t);
            ++b;
        }
        if (himages.size() > 1) {
            HConcat(himages, image_t);
        }
        vimages.push_back(image_t);
    }
    if (vimages.size() > 1) {
        VConcat(vimages, image_t);
    }
    return image_t;
}

void DLDataset::ToTensorPlane(const vector<int>& label, Tensor*& tensor)
{
    tensor = new Tensor({ vsize(classes_) });

    vector<float> lab(vsize(classes_), 0);
    for (int j = 0; j < vsize(label); ++j) {
        lab[label[j]] = 1;
    }
    std::copy(lab.data(), lab.data() + lab.size(), tensor->ptr);
}

void DLDataset::ProduceImageLabel(DatasetAugmentations& augs, Sample& elem)
{
    Image img = elem.LoadImage(ctype_, false);
    Tensor* label_tensor = nullptr, * image_tensor = nullptr;

    switch (task_) {
    case Task::classification:
    {
        // Read the label
        if (!split_[current_split_].no_label_) {
            auto label = elem.label_.value();
            ToTensorPlane(label, label_tensor);
        }
        // Apply chain of augmentations only to sample image
        augs.Apply(current_split_, img);
        ImageToTensor(img, image_tensor);

        queue_.Push(elem, image_tensor, label_tensor);
    }
    break;
    case Task::segmentation:
    {
        // Read the ground truth
        if (!split_[current_split_].no_label_) {
            Image gt = elem.LoadImage(ctype_gt_, true);
            // Apply chain of augmentations to sample image and corresponding ground truth
            augs.Apply(current_split_, img, gt);
            ImageToTensor(gt, label_tensor);
        }
        else {
            augs.Apply(current_split_, img);
        }
        ImageToTensor(img, image_tensor);

        queue_.Push(elem, image_tensor, label_tensor);
    }
    break;
    }
}

void DLDataset::InitTC(int split_index)
{
    auto& split_indexes = split_[split_index].samples_indices_;
    auto& drop_last = split_[split_index].drop_last_;
    auto samples_per_queue = vsize(split_indexes) / num_workers_;
    // if drop_last, subtract the number of samples that will be dropped out -> do not produce samples which are dropped
    auto exceeding_samples = vsize(split_indexes) % num_workers_ - (vsize(split_indexes) % batch_size_) * drop_last;

    // Set which are the indices of the samples managed by each thread
    // The i-th thread manage samples from start to end
    std::vector<ThreadCounters> split_tc;
    for (auto i = 0u; i < num_workers_; ++i) {
        auto start = samples_per_queue * i;
        auto end = start + samples_per_queue;
        if (i >= num_workers_ - 1) {
            // The last thread takes charge of exceeding samples
            end += exceeding_samples;
        }
        split_tc.push_back(ThreadCounters(start, end));
    }

    splits_tc_[split_index] = split_tc;
}

void DLDataset::ThreadFunc(int thread_index)
{
    auto& tc_of_current_split = splits_tc_[current_split_];
    DatasetAugmentations augs = augs_;
    while (tc_of_current_split[thread_index].counter_ < tc_of_current_split[thread_index].max_) {
        auto sample_index = split_[current_split_].samples_indices_[tc_of_current_split[thread_index].counter_];
        Sample& elem = samples_[sample_index];

        ProduceImageLabel(augs, elem);

        ++tc_of_current_split[thread_index].counter_;

        std::unique_lock<std::mutex> lock(active_mutex_);
        if (!active_) {
            return;
        }
    }
}

tuple<vector<Sample>, shared_ptr<Tensor>, shared_ptr<Tensor>> DLDataset::GetBatch()
{
    if (!active_) {
        cout << ECVL_WARNING_MSG << "You're trying to get a batch without starting the threads - you'll wait forever!" << endl;
    }

    ++current_batch_[current_split_];
    auto& s = split_[current_split_];
    auto tensors_shape = tensors_shape_;

    // Reduce batch size for the last batch in the split
    if (current_batch_[current_split_] == s.num_batches_) {
        tensors_shape.first[0] = s.last_batch_;
        if (!s.no_label_) {
            tensors_shape.second[0] = s.last_batch_;
        }
    }

    // If current split has no labels (e.g., test split could have no labels) set y as empty tensor
    tensors_shape.second = (s.no_label_) ? vector<int>{} : tensors_shape.second;

    Tensor* x = new Tensor(tensors_shape.first);
    Tensor* y = new Tensor(tensors_shape.second);

    Tensor* image;
    Tensor* label;

    const int batch_len = tensors_shape.first[0];
    vector<Sample> samples(batch_len);
    for (int i = 0; i < batch_len; ++i) {
        queue_.Pop(samples[i], image, label); // Consumer get samples from the queue
        image->unsqueeze_();
        memcpy(x->ptr + i * image->size, image->ptr, image->size * sizeof(float));
        delete image;

        if (label) {
            label->unsqueeze_();
            memcpy(y->ptr + i * label->size, label->ptr, label->size * sizeof(float));
            delete label;
        }
    }

    // wrap raw pointers in shared ptr, don't need to delete them
    shared_ptr<Tensor> x_shared(x);
    shared_ptr<Tensor> y_shared(y);

    return make_tuple(move(samples), move(x_shared), move(y_shared));
}

void DLDataset::Start(int split_index)
{
    if (active_) {
        ECVL_ERROR_START_ALREADY_ACTIVE
    }

    active_ = true;

    if (split_index != -1 && split_index != current_split_) {
        SetSplit(split_index);
    }

    // Check if current split has samples "to consume" -> reset_batch already called
    if (current_batch_.at(current_split_) > 0) {
        cerr << ECVL_WARNING_MSG << "`current_batch_` of the current split has not been reset. Did you miss a `ResetBatch()`?" << endl;
    }

    if (num_workers_ > 0) {
        for (auto i = 0u; i < num_workers_; ++i) {
            producers_.push_back(std::thread(&DLDataset::ThreadFunc, this, i));
        }
    }
    else {
        ECVL_ERROR_WORKERS_LESS_THAN_ONE
    }
}

void DLDataset::Stop()
{
    if (!active_) {
        ECVL_ERROR_STOP_ALREADY_END
    }

    {
        std::unique_lock<std::mutex> lock(active_mutex_);
        active_ = false;
        queue_.FreeLockedOnPush();
    }

    for (auto i = 0u; i < num_workers_; ++i) {
        producers_[i].join();
    }

    producers_.clear();
    queue_.Clear();
}

const int DLDataset::GetNumBatches(const any& split)
{
    auto it = GetSplitIt(split);
    return it->num_batches_;
}

void DLDataset::SetAugmentations(const DatasetAugmentations& da)
{
    augs_ = da;

    // Initialize resize_dims_ after that augmentations on the first image are performed
    Image tmp = samples_[0].LoadImage(ctype_);
    augs_.Apply(current_split_, tmp);
    auto y = tmp.channels_.find('y');
    auto x = tmp.channels_.find('x');
    assert(y != std::string::npos && x != std::string::npos);
    resize_dims_[0] = tmp.dims_[y];
    resize_dims_[1] = tmp.dims_[x];
}

void DLDataset::SetBatchSize(int bs)
{
    // check if the provided batch size is negative or greater than the current split size
    if (bs > 0 && bs < vsize(split_[current_split_].samples_indices_)) {
        int new_queue_size = static_cast<int>(queue_.Length() / batch_size_ * bs);
        batch_size_ = bs;
        tensors_shape_.first[0] = batch_size_;
        tensors_shape_.second[0] = batch_size_;
        queue_.SetSize(new_queue_size);
        for (auto& s : split_) {
            s.SetNumBatches(batch_size_);
            s.SetLastBatch(batch_size_);
        }
    }
    else {
        ECVL_ERROR_WRONG_PARAMS("bs in SetBatchSize")
    }
}
} // namespace ecvl