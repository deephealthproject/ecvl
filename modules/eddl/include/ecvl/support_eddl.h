/*
* ECVL - European Computer Vision Library
* Version: 0.3.4
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#ifndef ECVL_SUPPORT_EDDL_H_
#define ECVL_SUPPORT_EDDL_H_

#include "ecvl/augmentations.h"
#include "ecvl/core/filesystem.h"
#include "ecvl/core/image.h"
#include "ecvl/dataset_parser.h"

#include <eddl/apis/eddl.h>

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>

namespace ecvl
{
#define ECVL_ERROR_AUG_DOES_NOT_EXIST throw std::runtime_error(ECVL_ERROR_MSG "Augmentation for this split does not exist");
#define ECVL_ERROR_WORKERS_LESS_THAN_ONE throw std::runtime_error(ECVL_ERROR_MSG "Dataset workers must be at least one");

/** @brief Convert an ECVL Image into an EDDL Tensor.

Image must have 3 dimensions "xy[czo]" (in any order). \n
Output Tensor will be created with shape \f$C\f$ x \f$H\f$ x \f$W\f$. \n

@param[in] img Input ECVL Image.
@param[out] t Output EDDL Tensor. It is created inside the function.

*/
void ImageToTensor(const Image& img, Tensor*& t);

/** @brief Insert an ECVL Image into an EDDL Tensor.

This function is useful to insert into an EDDL Tensor more than one image, specifying how many images are already stored in the Tensor.
Image must have 3 dimensions "xy[czo]" (in any order). \n

@param[in] img Input ECVL Image.
@param[out] t Output EDDL Tensor. It must be created with the right dimensions before calling this function.
@param[in] offset How many images are already stored in the Tensor.

*/
void ImageToTensor(const Image& img, Tensor*& t, const int& offset);

/** @brief Convert an EDDL Tensor into an ECVL Image.

Tensor dimensions must be \f$C\f$ x \f$H\f$ x \f$W\f$ or \f$N\f$ x \f$C\f$ x \f$H\f$ x \f$W\f$, where: \n
\f$N\f$ = batch size \n
\f$C\f$ = channels \n
\f$H\f$ = height \n
\f$W\f$ = width

@param[in] t Input EDDL Tensor.
@param[out] img Output ECVL Image. It is a "xyo" with DataType::float32 and ColorType::none Image.

*/
void TensorToImage(const Tensor* t, Image& img);

/** @brief Convert an EDDL Tensor into an ECVL View.

Tensor dimensions must be \f$C\f$ x \f$H\f$ x \f$W\f$ or \f$N\f$ x \f$C\f$ x \f$H\f$ x \f$W\f$, where: \n
\f$N\f$ = batch size \n
\f$C\f$ = channels \n
\f$H\f$ = height \n
\f$W\f$ = width

@param[in] t Input EDDL Tensor.
@param[out] v Output ECVL View. It is a "xyo" with ColorType::none View.

*/
void TensorToView(const Tensor* t, View<DataType::float32>& v);

/** @brief Dataset Augmentations.

This class represent the augmentations which will be applied to each split.

This is just a shallow container for the Augmentations

@anchor DatasetAugmentations
*/
class DatasetAugmentations
{
    std::vector<std::shared_ptr<Augmentation>> augs_;
public:
    DatasetAugmentations(const std::vector<std::shared_ptr<Augmentation>>& augs) : augs_(augs) {}

    // This makes a deep copy of the Augmentations
    DatasetAugmentations(const DatasetAugmentations& other)
    {
        for (const auto& a : other.augs_) {
            augs_.emplace_back(a ? a->Clone() : nullptr);
        }
    }

    // Getters: YAGNI

    bool Apply(const int split, Image& img, const Image& gt = Image())
    {
        // check if the augs for split st are provided
        try {
            if (augs_.at(split)) {
                augs_[split]->Apply(img, gt);
                return true;
            }
            return false;
        }
        catch (const std::out_of_range) {
            ECVL_ERROR_AUG_DOES_NOT_EXIST
        }
    }

    bool Apply(SplitType st, Image& img, const Image& gt = Image())
    {
        return Apply(+st, img, gt); // Magic + operator
    }

    bool IsEmpty() const
    {
        return augs_.empty();
    }
};

/** @brief Class that manages the producers-consumer queue of samples.
* The queue stores pairs of image and label, pushing and popping them in an exclusive way.
* The queue also has a maximum size (`max_size_` attribute) to avoid memory overflows.

@anchor ProducersConsumerQueue
*/
class ProducersConsumerQueue
{
    std::condition_variable cond_notempty_;     /**< @brief Condition variable that wait if the queue is empty. */
    std::condition_variable cond_notfull_;      /**< @brief Condition variable that wait if the queue is full. */
    mutable std::mutex mutex_;                  /**< @brief Mutex to grant exclusive access to the queue. */
    std::queue<std::tuple<Sample, Tensor*, Tensor*>> cpq_;  /**< @brief Queue of samples, stored as tuple of Sample, Tensor* for the image and Tensor* for its label. */
    unsigned max_size_;                         /**< @brief Maximum size of the queue. */
    unsigned threshold_;                        /**< @brief Threshold from which restart to produce samples. If not specified, it's set to the half of maximum size. */

public:
    ProducersConsumerQueue() {}
    /**
    @param[in] mxsz Maximum size of the queue.
    */
    ProducersConsumerQueue(unsigned mxsz) : max_size_(mxsz), threshold_(max_size_ / 2) {}
    /**
    @param[in] mxsz Maximum size of the queue.
    @param[in] thresh Threshold from which restart to produce samples.
    */
    ProducersConsumerQueue(unsigned mxsz, unsigned thresh) : max_size_(mxsz), threshold_(thresh) {}

    /** @brief Push a sample in the queue.

    Take the lock of the queue and wait if the queue is full. Otherwise, push the tuple sample, image and label into the queue.

    @param[in] sample Sample to push in queue.
    @param[in] image Tensor* to the image to push in the queue.
    @param[in] label Tensor* to the label to push in the queue.
    */
    void Push(const Sample& sample, Tensor* const image, Tensor* const label)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_notfull_.wait(lock, [this]() { return cpq_.size() < max_size_; });
        cpq_.push(std::make_tuple(sample, image, label));
        cond_notempty_.notify_one();
    }

    /** @brief Pop a sample from the queue.

    Take the lock of the queue and wait if the queue is empty. Otherwise, pop a sample, an image and its label from the queue.
    If the queue size is still bigger than the half of the maximum size, don't notify the Push to avoid an always-full queue.

    @param[in] sample Sample to pop from the queue.
    @param[in] image Tensor* of the image to pop from the queue.
    @param[in] label Tensor* of the label to pop from the queue.
    */
    void Pop(Sample& sample, Tensor*& image, Tensor*& label)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_notempty_.wait(lock, [this]() { return !cpq_.empty(); });
        std::tie(sample, image, label) = cpq_.front();
        cpq_.pop();
        if (cpq_.size() < threshold_) {
            cond_notfull_.notify_all();
        }
    }

    /** @brief Check if the queue is full.

    @return true if the queue is full, false otherwise.
    */
    bool IsFull() const
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return cpq_.size() >= max_size_;
    }

    /** @brief Check if the queue is empty.

    @return true if the queue is empty, false otherwise.
    */
    bool IsEmpty() const
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return cpq_.empty();
    }

    /** @brief Calculate the current size of the queue.

    @return the current size of the queue.
    */
    size_t Length() const
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return cpq_.size();
    }

    /** @brief Set the maximum size of the queue and optionally the threshold from which restart to produce samples.

    @param[in] max_size maximum size of the queue.
    @param[in] thresh threshold from which restart to produce samples. If not specified, it's set to the half of maximum size.
    */
    void SetSize(int max_size, int thresh = -1)
    {
        max_size_ = max_size;
        threshold_ = thresh != -1 ? thresh : max_size / 2;
    }

    void Clear()
    {
        std::unique_lock<std::mutex> lock(mutex_);

        // Remove residual samples and delete data
        while (!cpq_.empty()) {
            cpq_.pop();
        }
    }
};

/** @brief Class representing the thread counters.

Each thread has its own indices to manage. The split samples have been assigned to several threads which manage them separately.

@anchor ThreadCounters
*/
class ThreadCounters
{
public:
    int counter_;   /**< @brief Index of the sample currently used by the thread. */
    int min_, max_; /**< @brief Indices of samples managed by the thread in the interval [min_, max_). */

    ThreadCounters(int min, int max) : counter_{ min }, min_{ min }, max_{ max } {}
    ThreadCounters(int counter, int min, int max) : counter_{ counter }, min_{ min }, max_{ max } {}
    void Reset() { counter_ = min_; }  /**< @brief Reset the thread counter to its minimum value. */
};

/** @brief DeepHealth Deep Learning Dataset.

This class extends the DeepHealth Dataset with Deep Learning specific members.

@anchor DLDataset
*/
class DLDataset : public Dataset
{
    const unsigned processor_count_ = std::thread::hardware_concurrency();
protected:
    int batch_size_;                            /**< @brief Size of each dataset mini batch. */
    std::vector<int> current_batch_;            /**< @brief Number of batches already loaded for each split. */
    ColorType ctype_;                           /**< @brief ecvl::ColorType of the Dataset images. */
    ColorType ctype_gt_;                        /**< @brief ecvl::ColorType of the Dataset ground truth images. */
    DatasetAugmentations augs_;                 /**< @brief ecvl::DatasetAugmentations to be applied to the Dataset images (and ground truth if exist) for each split. */
    unsigned num_workers_;                      /**< @brief Number of parallel workers. */
    ProducersConsumerQueue queue_;              /**< @brief Producers-consumer queue of the dataset. */
    std::pair< std::vector<int>, std::vector<int>> tensors_shape_; /**< @brief Shape of sample and label tensors. */
    std::vector<std::vector<ThreadCounters>> splits_tc_; /**< @brief Each dataset split has its own vector of threads, each of which has its counters: <counter,min,max>. */
    std::vector<std::thread> producers_;        /**< @brief Vector of threads representing the samples producers. */
    bool active_ = false;                       /**< @brief Whether the threads have already been launched or not. */
    std::mutex active_mutex_;                   /**< @brief Mutex for active_ variable. */
    static std::default_random_engine re_;      /**< @brief Engine used for random number generation. */

    /** @brief Set which are the indices of the samples managed by each thread.

    @param[in] split_index index of the split to initialize.
    */
    void InitTC(int split_index);

    /** @brief Set internal EDDL tensors shape. */
    void SetTensorsShape()
    {
        switch (task_) {
        case Task::classification:
            tensors_shape_ = std::make_pair<std::vector<int>, std::vector<int>>(
                { batch_size_, n_channels_, resize_dims_[0], resize_dims_[1] },
                { batch_size_, vsize(classes_) });
            break;
        case Task::segmentation:
            tensors_shape_ = std::make_pair<std::vector<int>, std::vector<int>>(
                { batch_size_, n_channels_, resize_dims_[0], resize_dims_[1] },
                { batch_size_, n_channels_gt_, resize_dims_[0], resize_dims_[1] });
            break;
        }
    }

public:
    int n_channels_;                            /**< @brief Number of channels of the images. */
    int n_channels_gt_ = -1;                    /**< @brief Number of channels of the ground truth images. */
    std::vector<int> resize_dims_;              /**< @brief Dimensions (HxW) to which Dataset images must be resized. */

    /**
    @param[in] filename Path to the Dataset file.
    @param[in] batch_size Size of each dataset mini batch.
    @param[in] augs Array with DatasetAugmentations to be applied to the Dataset images (and ground truth if exists) for each split. If no augmentation is required nullptr has to be passed.
    @param[in] ctype ecvl::ColorType of the Dataset images. Default is RGB.
    @param[in] ctype_gt ecvl::ColorType of the Dataset ground truth images. Default is GRAY.
    @param[in] num_workers Number of parallel threads spawned.
    @param[in] queue_ratio_size The producers-consumer queue will have a maximum size equal to \f$batch\_size \times queue\_ratio\_size \times num\_workers\f$.
    @param[in] drop_last For each split, whether to drop the last samples that don't fit the batch size or not. The vector dimensions must match the number of splits.
    @param[in] verify If true, a list of all the images in the Dataset file which don't exist is printed with an ECVL_WARNING_MSG.
    */
    DLDataset(const filesystem::path& filename,
        const int batch_size,
        const DatasetAugmentations& augs,
        const ColorType ctype = ColorType::RGB,
        const ColorType ctype_gt = ColorType::GRAY,
        const unsigned num_workers = 1,
        const double queue_ratio_size = 1.,
        const std::vector<bool>& drop_last = {},
        bool verify = false) :

        Dataset{ filename, verify },
        batch_size_{ batch_size },
        augs_(augs),
        num_workers_{ std::min(num_workers, processor_count_) },
        ctype_{ ctype },
        ctype_gt_{ ctype_gt },
        queue_{ static_cast<unsigned>(batch_size * queue_ratio_size * std::min(num_workers, processor_count_)) }
    {
        // resize current_batch_ to the number of splits and initialize it with 0
        current_batch_.resize(split_.size(), 0);

        // Initialize n_channels_
        Image tmp = samples_[0].LoadImage(ctype);
        n_channels_ = tmp.Channels();

        if (!split_.empty()) {
            current_split_ = 0;
            // Initialize resize_dims_ after that augmentations on the first image are performed
            if (augs_.IsEmpty()) {
                std::cout << ECVL_WARNING_MSG << "Augmentations are empty!" << std::endl;
            }
            else {
                while (!augs_.Apply(current_split_, tmp)) {
                    ++current_split_;
                }
            }
            auto y = tmp.channels_.find('y');
            auto x = tmp.channels_.find('x');
            assert(y != std::string::npos && x != std::string::npos);
            resize_dims_.insert(resize_dims_.begin(), { tmp.dims_[y],tmp.dims_[x] });

            // Initialize n_channels_gt_ if exists
            if (samples_[0].label_path_ != nullopt) {
                n_channels_gt_ = samples_[0].LoadImage(ctype_gt_, true).Channels();
            }
        }
        else {
            std::cout << ECVL_WARNING_MSG << "Missing splits in the dataset file." << std::endl;
        }

        // Set drop_last parameter for each split
        if (!drop_last.empty()) {
            if (vsize(drop_last) == vsize(split_)) {
                for (int i = 0; i < vsize(drop_last); ++i) {
                    split_[i].drop_last_ = drop_last[i];
                }
            }
            else {
                std::cout << ECVL_WARNING_MSG << "drop_last is not empty but the provided size is different from the size of the splits. The default value 'false' is set for all the splits" << std::endl;
            }
        }

        // Initialize num_batches, last_batch and the ThreadCounters for each split
        auto s_index = 0;
        splits_tc_ = std::vector<std::vector<ThreadCounters>>(vsize(split_));
        for (auto& s : split_) {
            s.SetNumBatches(batch_size_);
            s.SetLastBatch(batch_size_);

            InitTC(s_index);
            ++s_index;
        }

        SetTensorsShape();
    }

    /** @brief Reset the batch counter and optionally shuffle samples indices of the specified split.

    If no split is provided or an illegal value is provided, the current split is reset.
    @param[in] split_index index, name or SplitType of the split to reset.
    @param[in] shuffle boolean which indicates whether to shuffle the split samples indices or not.
    */
    void ResetBatch(const ecvl::any& split = -1, bool shuffle = false);

    /** @brief Reset the batch counter of each split and optionally shuffle samples indices (within each split).

    @param[in] shuffle boolean which indicates whether to shuffle the samples indices or not.
    */
    void ResetAllBatches(bool shuffle = false);

    /** @brief Load a batch into _images_ and _labels_ `tensor`.

    @param[out] images `tensor` which stores the batch of images.
    @param[out] labels `tensor` which stores the batch of labels.
    */
    void LoadBatch(Tensor*& images, Tensor*& labels);

    /** @brief Load a batch into _images_ `tensor`. Useful for tests set when you don't have labels.

    @param[out] images `tensor` which stores the batch of images.
    */
    void LoadBatch(Tensor*& images);

    /** @brief Set a fixed seed for the random generated values. Useful to reproduce experiments with same shuffling during training.

    @param[in] seed Value of the seed for the random engine.
    */
    static void SetSplitSeed(unsigned seed) { re_.seed(seed); }

    /** @brief Set a new batch size inside the dataset.

    Notice that this will not affect the EDDL network batch size, that it has to be changed too.
    @param[in] bs Value to set for the batch size.
    */
    void SetBatchSize(int bs);

    /** @brief Load a sample and its label, and push them to the producers-consumer queue.

    @param[in] elem Sample to load and push to the queue.

    @anchor ProduceImageLabel
    */
    virtual void ProduceImageLabel(DatasetAugmentations& augs, Sample& elem);

    /** @brief Function called when the thread are spawned.

    @ref ProduceImageLabel is called for each sample under the competence of the thread.

    @param[in] thread_index index of the thread.
    */
    void ThreadFunc(int thread_index);

    /** @brief Pop batch_size samples from the queue and copy them into EDDL tensors.

    @return tuples of Samples and EDDL Tensors, the first with the image and the second with the label.
    */
    std::tuple<std::vector<Sample>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> GetBatch();

    /** @brief Spawn num_workers thread.

    @param[in] split_index Index of the split to use in the GetBatch function. If not specified, current split is used.
    */
    void Start(int split_index = -1);

    /** @brief Join all the threads. */
    void Stop();

    /** @brief Get the current size of the producers-consumer queue of the dataset.

    @return Size of the producers-consumer queue of the dataset.
    */
    auto GetQueueSize() const { return queue_.Length(); };

    /** @brief Set the dataset augmentations.

    @param[in] da @ref DatasetAugmentations to set.
    */
    void SetAugmentations(const DatasetAugmentations& da);

    /** @brief Get the number of batches of the specified split.

    If no split is provided or an illegal value is provided, the number of batches of the current split is returned.
    @param[in] split index, name or ecvl::SplitType representing the split from which to get the number of batches.
    @return number of batches of the specified split.
    */
    const int GetNumBatches(const ecvl::any& split = -1);

    /** @brief Convert the sample labels in a one-hot encoded tensor and copy it to the batch tensor.

    @param[in] label vector of the sample labels
    @param[out] tensor EDDL Tensor in which to copy the labels (dimensions: [batch_size, num_classes])
    */
    void ToTensorPlane(const std::vector<int>& label, Tensor*& tensor);

    /** @brief Change the number of workers.

    @param[in] num_workers Number of threads/workers that will be spawned.
    */
    void SetWorkers(const unsigned num_workers)
    {
        if (num_workers < 0) {
            ECVL_ERROR_WORKERS_LESS_THAN_ONE
        }

        num_workers_ = std::min(num_workers, processor_count_);
        for (int i = 0; i < vsize(split_); ++i) {
            InitTC(i);
        }
    }

    /** @brief Change the number of channels of the Image produced by ECVL and update the internal EDDL tensors shape accordingly.
    Useful for custom data loading.

    @param[in] n_channels Number of channels of input Image.
    @param[in] n_channels_gt Number of channels of ground truth.
    */
    void SetNumChannels(const int n_channels, const int n_channels_gt = 1)
    {
        n_channels_ = n_channels;
        n_channels_gt_ = n_channels_gt;
        SetTensorsShape();
    }
};

/** @brief Make a grid of images from a EDDL Tensor.

Return a grid of Image from a EDDL Tensor.

@param[in] t Input EDDL Tensor of shape (B x C x H x W).
@param[in] cols Number of images displayed in each row of the grid.
@param[in] normalize If true, shift the image to the range [0,1].

@return Image that contains the grid of images
*/
Image MakeGrid(Tensor*& t, int cols = 8, bool normalize = false);

/** @example example_ecvl_eddl.cpp
 Example of using ECVL with EDDL.
*/
} // namespace ecvl

#endif // ECVL_SUPPORT_EDDL_H_
