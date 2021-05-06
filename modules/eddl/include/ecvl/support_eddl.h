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

#include <mutex>

namespace ecvl
{
#define ECVL_ERROR_AUG_DOES_NOT_EXIST throw std::runtime_error(ECVL_ERROR_MSG "Augmentation for this split does not exist");

/** @brief Dataset Augmentations.

This class represent the augmentations which will be applied to each split.

This is just a shallow container for the Augmentations

@anchor DatasetAugmentations
*/
class DatasetAugmentations
{
    std::vector<shared_ptr<Augmentation>> augs_;
public:
    DatasetAugmentations(const std::vector<shared_ptr<Augmentation>>& augs = { nullptr, nullptr, nullptr }) : augs_(augs) {}

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
};

/** @brief DeepHealth Deep Learning Dataset.

This class extends the DeepHealth Dataset with Deep Learning specific members.

@anchor DLDataset
*/
class DLDataset : public Dataset
{
public:
    int batch_size_; /**< @brief Size of each dataset mini batch. */
    int n_channels_; /**< @brief Number of channels of the images. */
    int n_channels_gt_ = -1; /**< @brief Number of channels of the ground truth images. */
    std::vector<int> resize_dims_; /**< @brief Dimensions (HxW) to which Dataset images must be resized. */
    std::vector<int> current_batch_; /**< @brief Number of batches already loaded for each split. */
    ColorType ctype_; /**< @brief ecvl::ColorType of the Dataset images. */
    ColorType ctype_gt_; /**< @brief ecvl::ColorType of the Dataset ground truth images. */
    DatasetAugmentations augs_; /**< @brief ecvl::DatasetAugmentations to be applied to the Dataset images (and ground truth if exist) for each split. */
    std::mutex  mutex_current_batch_; /**< @brief std::mutex to add exclusive access to attribute current_batch_. */
    static std::default_random_engine re_;

    /**
    @param[in] filename Path to the Dataset file.
    @param[in] batch_size Size of each dataset mini batch.
    @param[in] augs Array with three DatasetAugmentations (training, validation and test) to be applied to the Dataset images (and ground truth if exists) for each split.
                    If no augmentation is required or the split doesn't exist, nullptr has to be passed.
    @param[in] ctype ecvl::ColorType of the Dataset images.
    @param[in] ctype_gt ecvl::ColorType of the Dataset ground truth images.
    @param[in] verify If true, a list of all the images in the Dataset file which don't exist is printed with an ECVL_WARNING_MSG.
    */
    DLDataset(const filesystem::path& filename,
        const int batch_size,
        DatasetAugmentations augs = DatasetAugmentations(),
        ColorType ctype = ColorType::BGR,
        ColorType ctype_gt = ColorType::GRAY,
        bool verify = false) :

        Dataset{ filename, verify },
        batch_size_{ batch_size },
        augs_(std::move(augs)),
        ctype_{ ctype },
        ctype_gt_{ ctype_gt }
    {
        // resize current_batch_ to the number of splits and initialize it with 0
        current_batch_.resize(split_.size(), 0);

        // Initialize n_channels_
        Image tmp = samples_[0].LoadImage(ctype);
        n_channels_ = tmp.Channels();

        if (!split_.empty()) {
            current_split_ = 0;
            // Initialize resize_dims_ after that augmentations on the first image are performed
            augs_.Apply(current_split_, tmp);
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
            cout << ECVL_WARNING_MSG << "Missing splits in the dataset file." << endl;
        }
    }

    /** @brief Reset the batch counter and optionally shuffle samples indices of the specified split.

    If no split is provided (i.e. it is provided a value less than 0), the current split is reset.
    @param[in] split_index index of the split to reset.
    @param[in] reshuffle boolean which indicates whether to shuffle the split samples indices or not.
    */
    void ResetBatch(int split_index = -1, bool shuffle = false);

    /** @brief Reset the batch counter and optionally shuffle samples indices of the specified split.

    @param[in] split_name name of the split to reset.
    @param[in] reshuffle boolean which indicates whether to shuffle the split samples indices or not.
    */
    void ResetBatch(std::string split_name, bool shuffle = false);

    /** @brief Reset the batch counter and optionally shuffle samples indices of the specified split.

    @param[in] split_type SplitType of the split to reset.
    @param[in] reshuffle boolean which indicates whether to shuffle the split samples indices or not.
    */
    void ResetBatch(SplitType split_type, bool shuffle = false);

    /** @brief Reset the batch counter of each split and optionally shuffle samples indices (within each split).

    @param[in] reshuffle boolean which indicates whether to shuffle the samples indices or not.
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
    static void SetSplitSeed(unsigned seed)
    {
        re_.seed(seed);
    }

    /** @brief Set a new batch size inside the dataset.

    Notice that this will not affect the EDDL network batch size, that it has to be changed too.
    @param[in] bs Value to set for the batch size.
    */
    void SetBatchSize(int bs);
};

/** @brief Convert an EDDL Tensor into an ECVL Image.

Tensor dimensions must be \f$C\f$ x \f$H\f$ x \f$W\f$ or \f$N\f$ x \f$C\f$ x \f$H\f$ x \f$W\f$, where: \n
\f$N\f$ = batch size \n
\f$C\f$ = channels \n
\f$H\f$ = height \n
\f$W\f$ = width

@param[in] t Input EDDL Tensor.
@param[out] img Output ECVL Image. It is a "xyo" with DataType::float32 and ColorType::none Image.

*/
void TensorToImage(Tensor*& t, Image& img);

/** @brief Convert an EDDL Tensor into an ECVL View.

Tensor dimensions must be \f$C\f$ x \f$H\f$ x \f$W\f$ or \f$N\f$ x \f$C\f$ x \f$H\f$ x \f$W\f$, where: \n
\f$N\f$ = batch size \n
\f$C\f$ = channels \n
\f$H\f$ = height \n
\f$W\f$ = width

@param[in] t Input EDDL Tensor.
@param[out] v Output ECVL View. It is a "xyo" with ColorType::none View.

*/
void TensorToView(Tensor*& t, View<DataType::float32>& v);

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

/** @brief Make a grid of images from a EDDL Tensor.

Return a grid of Image from a EDDL Tensor.

@param[in] img Input EDDL Tensor of shape (B x C x H x W).
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
