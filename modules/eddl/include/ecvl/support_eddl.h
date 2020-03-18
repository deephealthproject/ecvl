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

#ifndef ECVL_SUPPORT_EDDL_H_
#define ECVL_SUPPORT_EDDL_H_

#include <eddl/apis/eddl.h>
#include <eddl/apis/eddlT.h>
#include "ecvl/augmentations.h"
#include "ecvl/core/image.h"
#include "ecvl/dataset_parser.h"

namespace ecvl
{
/** @brief Dataset Augmentations.

This class represent the augmentations which will be applied to each split.

@anchor DatasetAugmentations
*/
class DatasetAugmentations {
public:
    std::array<unique_ptr<Augmentation>, 3> augs_;

    DatasetAugmentations(std::array<unique_ptr<Augmentation>, 3> augs = { nullptr,nullptr,nullptr }) : augs_{ std::move(augs) } {}

    void Apply(SplitType st, Image& img, const Image& gt = Image())
    {
        if (augs_[+st]) { // Magic + operator
            augs_[+st]->Apply(img, gt);
        }
    }
};

/** @brief DeepHealth Deep Learning Dataset.

This class extends the DeepHealth Dataset with Deep Learning specific members.

@anchor DLDataset
*/
class DLDataset : public Dataset {
public:
    int batch_size_; /**< @brief Size of each dataset mini batch. */
    int n_channels_; /**< @brief Number of channels of the images. */
    int n_channels_gt_; /**< @brief Number of channels of the ground truth images. */
    SplitType current_split_ = SplitType::training; /**< @brief Current split from which images are loaded. */
    std::vector<int> resize_dims_; /**< @brief Dimensions (HxW) to which Dataset images must be resized. */
    std::array<int, 3> current_batch_ = { 0,0,0 }; /**< @brief Number of batches already loaded for each split. */
    ColorType ctype_; /**< @brief ecvl::ColorType of the Dataset images. */
    ColorType ctype_gt_; /**< @brief ecvl::ColorType of the Dataset ground truth images. */
    DatasetAugmentations augs_; /**< @brief ecvl::DatasetAugmentations to be applied to the Dataset images (and ground truth if exist) for each split. */

    /**
    @param[in] filename Path to the Dataset file.
    @param[in] batch_size Size of each dataset mini batch.
    @param[in] augs Array with three DatasetAugmentations (training, validation and test) to be applied to the Dataset images (and ground truth if exists) for each split.
                    If no augmentation is required or the split doesn't exist, nullptr has to be passed.
    @param[in] ctype ecvl::ColorType of the Dataset images.
    @param[in] ctype_gt ecvl::ColorType of the Dataset ground truth images.
    @param[in] verify If true, a list of all the images in the Dataset file which don't exist is printed with an ECVL_WARNING_MSG.
    */
    DLDataset(const std::filesystem::path& filename,
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
        Image tmp = this->samples_[0].LoadImage(ctype);
        // Initialize resize_dims_ after that augmentations on images are performed
        augs_.Apply(current_split_, tmp);
        int y = tmp.channels_.find('y');
        int x = tmp.channels_.find('x');
        assert(y != std::string::npos && x != std::string::npos);
        resize_dims_.insert(resize_dims_.begin(), { tmp.dims_[y],tmp.dims_[x] });

        // Initialize n_channels_
        n_channels_ = tmp.Channels();
        // Initialize n_channels_gt_ if exists
        if (this->split_.training_.size() > 0) {
            if (this->samples_[this->split_.training_[0]].label_path_.has_value()) {
                n_channels_gt_ = this->samples_[this->split_.training_[0]].LoadImage(ctype_gt_, true).Channels();
            }
        }
    }

    /** @brief Returns the image indexes of the current Split.
    @return vector of image indexes of the Split in use.
    */
    std::vector<int>& GetSplit();

    /** @brief Returns the image indexes of the requested Split.
    @param[in] split ecvl::SplitType representing the Split to get ("training", "validation", or "test").
    @return vector of image indexes of the requested Split.
    */
    std::vector<int>& GetSplit(const SplitType& split);

    /** @brief Reset the batch counter of the current Split. */
    void ResetCurrentBatch();

    /** @brief Reset the batch counter of each Split. */
    void ResetAllBatches();

    /** @brief Set the current Split.
    @param[in] split ecvl::SplitType representing the Split to set ("training", "validation", or "test").
    */
    void SetSplit(const SplitType& split);

    /** @brief Load a batch into _images_ and _labels_ `tensor`.
    @param[out] images `tensor` which stores the batch of images.
    @param[out] labels `tensor` which stores the batch of labels.
    */
    void LoadBatch(tensor& images, tensor& labels);

    /** @brief Load a batch into _images_ `tensor`. Useful for tests set when you don't have labels.
    @param[out] images `tensor` which stores the batch of images.
    */
    void LoadBatch(tensor& images);
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
void TensorToImage(tensor& t, Image& img);

/** @brief Convert an EDDL Tensor into an ECVL View.

Tensor dimensions must be \f$C\f$ x \f$H\f$ x \f$W\f$ or \f$N\f$ x \f$C\f$ x \f$H\f$ x \f$W\f$, where: \n
\f$N\f$ = batch size \n
\f$C\f$ = channels \n
\f$H\f$ = height \n
\f$W\f$ = width

@param[in] t Input EDDL Tensor.
@param[out] v Output ECVL View. It is a "xyo" with ColorType::none View.

*/
void TensorToView(tensor& t, View<DataType::float32>& v);

/** @brief Convert an ECVL Image into an EDDL Tensor.

Image must have 3 dimensions "xy[czo]" (in any order). \n
Output Tensor will be created with shape \f$C\f$ x \f$H\f$ x \f$W\f$. \n

@param[in] img Input ECVL Image.
@param[out] t Output EDDL Tensor. It is created inside the function.

*/
void ImageToTensor(const Image& img, tensor& t);

/** @brief Insert an ECVL Image into an EDDL Tensor.

This function is useful to insert into an EDDL Tensor more than one image, specifying how many images are already stored in the Tensor.
Image must have 3 dimensions "xy[czo]" (in any order). \n

@param[in] img Input ECVL Image.
@param[out] t Output EDDL Tensor. It must be created with the right dimensions before calling this function.
@param[in] offset How many images are already stored in the Tensor.

*/
void ImageToTensor(const Image& img, tensor& t, const int& offset);

/** @example example_ecvl_eddl.cpp
 Example of using ECVL with EDDL.
*/
} // namespace ecvl

#endif // ECVL_SUPPORT_EDDL_H_