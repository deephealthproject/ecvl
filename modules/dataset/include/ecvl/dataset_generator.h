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

#ifndef ECVL_DATASET_GENERATOR_H_
#define ECVL_DATASET_GENERATOR_H_

#include <ecvl/dataset_parser.h>
#include <filesystem>

namespace ecvl
{
/** @brief Abstract class which fill the dataset object with name and description, features common to all types of datasets.

@anchor GenerateDataset
*/
class GenerateDataset
{
public:
    const std::filesystem::path dataset_root_directory_;    /**< @brief path containing the root directory of the dataset */
    std::vector<std::string> splits_;                       /**< @brief vector containing the splits found in the dataset directory, if present */
    std::vector<int> num_samples_;                          /**< @brief vector containing the number of samples for each split*/
    ecvl::Dataset d_;                                       /**< @brief Dataset object to fill */

    /** @brief GenerateDataset constructor

    @param[in] dataset_root_directory path containing the root directory of the dataset.
    */
    GenerateDataset(const std::filesystem::path& dataset_root_directory) :
        dataset_root_directory_(dataset_root_directory)
    {
        for (auto& p : std::filesystem::directory_iterator(dataset_root_directory_)) {
            std::string tmp = p.path().stem().string();

            // Check if split folders exist
            if (tmp == "training" || tmp == "validation" || tmp == "test") {
                splits_.emplace_back(tmp);
            }
        }
        num_samples_.resize(splits_.size());

        d_.name_ = dataset_root_directory_.stem().string();
        d_.description_ = dataset_root_directory_.stem().string();
    }

    virtual ~GenerateDataset() = default;

    /** @brief Call LoadSplitImages and load the splits with indexes of corresponding images.

    If there aren't splits folders, only the list of images and corresponding labels will be loaded.
    */
    void LoadImagesAndSplits();

    /** @brief Return the Dataset object obtained from the directory structure.

    @return Dataset obtained from the directory structure.
    */
    Dataset GetDataset() { return d_; }

    /** @brief Load the path of images and labels of the specified split.

    @param[in] split directory name of the split that we are considering.
    @return The number of samples of the split.
    */
    virtual int LoadSplitImages(const std::filesystem::path& split) = 0;
};

/** @brief Generate an ecvl::Dataset from a directory tree for a segmentation task.

Assumes a directory structure where a top-level directory can have subdirectories
named "training", "validation" and "test" (possibly not all present), each of
which can have images and ground truth in different subdirectories (named "images" and "ground_truth")
or in the same directory. If the ground truth images have a particular suffix or a different extension
(necessary if they are in the same directory as the images) it's necessary to specify it in the constructor.
For more detailed information about the supported directory structure check https://github.com/deephealthproject/ecvl/wiki/ECVL-Dataset-Generator.

@anchor GenerateSegmentationDataset
*/
class GenerateSegmentationDataset : public GenerateDataset
{
public:
    std::filesystem::path suffix_;  /**< @brief path containing the suffix or extension of ground truth images */
    std::filesystem::path gt_name_; /**< @brief path containing the ground truth name for images that share the same ground truth */

    /** @brief GenerateSegmentationDataset constructor

    @param[in] dataset_root_directory path containing the root directory of the dataset.
    @param[in] suffix suffix or extension of ground truth images, necessary if it's different from corresponding images (e.g., "_segmentation.png" or ".png").
    @param[in] gt_name name of the ground truth for images that share the same ground truth. 
                       All images in the split (if available) which don't have their own ground truth will use this one.
    */
    GenerateSegmentationDataset(const std::filesystem::path& dataset_root_directory, std::filesystem::path suffix = "", std::filesystem::path gt_name = "") :
        GenerateDataset{ dataset_root_directory },
        suffix_{ suffix },
        gt_name_{ gt_name }
    {
        LoadImagesAndSplits();
    }

    virtual int LoadSplitImages(const std::filesystem::path& split) override;
};

/** @brief Generate an ecvl::Dataset from a directory tree for a classification task.

Assumes a directory structure where a top-level directory can have subdirectories
named "training", "validation" and "test" (possibly not all present), each of
which has in turn one subdirectory for each class, containing the images for that class.
For more detailed information about the supported directory structure check https://github.com/deephealthproject/ecvl/wiki/ECVL-Dataset-Generator.

@anchor GenerateClassificationDataset
*/
class GenerateClassificationDataset : public GenerateDataset
{
public:

    /** @brief GenerateClassificationDataset constructor.

    All the splits must have a directory for each class. If there aren't samples of that class, the directory has to be empty.

    @param[in] dataset_root_directory path containing the root directory of the dataset.
    */
    GenerateClassificationDataset(const std::filesystem::path& dataset_root_directory) : GenerateDataset{ dataset_root_directory }
    {
        std::filesystem::path tmp;

        if (splits_.empty()) {
            tmp = dataset_root_directory_;
        }
        else {
            // use the first split available to list all the classes
            tmp = dataset_root_directory_ / splits_[0];
        }

        for (auto& p : std::filesystem::directory_iterator(tmp)) {
            if (std::filesystem::is_directory(p.path())) {
                d_.classes_.push_back(p.path().stem().string());
            }
        }

        LoadImagesAndSplits();
    }

    virtual int LoadSplitImages(const std::filesystem::path& split) override;
};
} // namespace ecvl

#endif // ECVL_DATASET_GENERATOR_H_