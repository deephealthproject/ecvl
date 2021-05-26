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

#ifndef ECVL_DATASET_PARSER_H_
#define ECVL_DATASET_PARSER_H_

#include "ecvl/core.h"
#include "ecvl/core/any.h"
#include "ecvl/core/filesystem.h"
#include "ecvl/core/optional.h"

#include <iostream>
#include <iterator>
#include <map>
#include <vector>
#include <regex>

#include "yaml-cpp/yaml.h"

// This allows to define strongly typed enums and convert them to int with just a + in front
#define UNSIGNED_ENUM_CLASS(name, ...) enum class name : unsigned { __VA_ARGS__ };\
inline constexpr unsigned operator+ (name const val) { return static_cast<unsigned>(val); }

namespace ecvl
{
/** @brief Enum class representing the Dataset supported splits.

@anchor SplitType
*/
UNSIGNED_ENUM_CLASS(SplitType, training, validation, test)

/** @brief Enum class representing allowed tasks for the ECVL Dataset.

@anchor Task
 */
enum class Task
{
    classification,
    segmentation,
};

/** @brief Sample image in a dataset.

This class provides the information to describe a dataset sample.
`label_` and `label_path_` are mutually exclusive.
@anchor Sample
*/
class Sample
{
public:
    std::vector<filesystem::path> location_;        /**< @brief Absolute path of the sample. */
    optional<std::vector<int>> label_;              /**< @brief Vector of sample labels. */
    optional<filesystem::path> label_path_;         /**< @brief Absolute path of sample ground truth. */
    optional<std::map<int, std::string>> values_;   /**< @brief Map (`map<feature-index,feature-value>`) which stores the features of a sample. */
    std::vector<int> size_;                         /**< @brief Original x and y dimensions of the sample */

    /** @brief Return an Image of the dataset.

    The LoadImage() function opens the sample image, from `location_` or `label_path_` depending on `is_gt` parameter.

    @param[in] ctype ecvl::ColorType of the returned Image.
    @param[in] is_gt Whether to load the sample image or its ground truth.

    @return Image containing the loaded sample.
    */
    ecvl::Image LoadImage(ecvl::ColorType ctype = ecvl::ColorType::RGB, const bool& is_gt = false);
};

/** @brief Split of a dataset.
This class provides the name of the split and the indices of the samples that belong to this split.
It optionally provides the split type if the split name is one of training, validation or test.
@anchor Split
*/
class Split
{
public:
    std::string split_name_;                /**< @brief Name of the split. */
    optional<SplitType> split_type_;        /**< @brief If the split is training, validation or test the corresponding SpitType is provided. */
    std::vector<int> samples_indices_;      /**< @brief Vector containing samples indices of the split. */
    bool drop_last_ = false;                /**< @brief Whether to drop elements that don't fit batch size or not. */
    int num_batches_;                       /**< @brief Number of batches of this split. */
    int last_batch_;                        /**< @brief Dimension of the last batch of this split. */
    bool no_label_ = false;                 /**< @brief Whether the split has samples with labels or not. */

    Split() {}

    /**
    @param[in] split_name Name of the split.
    @param[in] samples_indices Vector containing samples indices of the split.
    */
    Split(const std::string& split_name, const std::vector<int>& samples_indices) : split_name_{ split_name }, samples_indices_{ samples_indices }
    {
        if (split_name_ == "training") split_type_ = SplitType::training;
        else if (split_name_ == "validation") split_type_ = SplitType::validation;
        else if (split_name_ == "test") split_type_ = SplitType::test;
    }

    void SetNumBatches(int batch_size)
    {
        num_batches_ = drop_last_ ? vsize(samples_indices_) / batch_size : (vsize(samples_indices_) + batch_size - 1) / batch_size;
    }

    void SetLastBatch(int batch_size)
    {
        // last batch is the remainder of the number of samples of the split divided by the batch size.
        // if drop last is true or the remainder is 0, last batch is equal to the batch size.
        auto value = vsize(samples_indices_) % batch_size;
        last_batch_ = drop_last_ ? batch_size : (value == 0 ? batch_size : value);
    }
};

/** @brief DeepHealth Dataset.

This class implements the DeepHealth Dataset Format (https://github.com/deephealthproject/ecvl/wiki/DeepHealth-Toolkit-Dataset-Format).

@anchor Dataset
*/
class Dataset
{
    std::map<std::string, int> features_map_;
    void DecodeImages(const YAML::Node& node, const filesystem::path& root_path, bool verify);
    void FindLabel(Sample& sample, const YAML::Node& n);
protected:
    std::vector<ecvl::Split>::iterator GetSplitIt(ecvl::any split);
    int GetSplitIndex(ecvl::any split);
public:
    std::string name_ = "DeepHealth dataset";                               /**< @brief Name of the Dataset. */
    std::string description_ = "This is the DeepHealth example dataset!";   /**< @brief Description of the Dataset. */
    std::vector<std::string> classes_;                                      /**< @brief Vector with all the classes available in the Dataset. */
    std::vector<std::string> features_;                                     /**< @brief Vector with all the features available in the Dataset. */
    std::vector<Sample> samples_;                                           /**< @brief Vector containing all the Dataset samples. See @ref Sample. */
    std::vector<Split> split_;                                              /**< @brief Splits of the Dataset. See @ref Split. */
    int current_split_ = -1;                                                /**< @brief Current split from which images are loaded. */
    Task task_;                                                             /**< @brief Task of the dataset. */

    Dataset() {}

    /**
    @param[in] filename Path to the Dataset file.
    @param[in] verify Whether to log the non-existence of a dataset sample location or not.
    */
    Dataset(const filesystem::path& filename, bool verify = false);

    /* Destructor */
    virtual ~Dataset() {}

    /** @brief Returns the image indexes of the requested split.

    If no split is provided or an illegal value is provided, the current split is returned.
    @param[in] split index, name or ecvl::SplitType representing the split to get.
    @return vector of image indexes of the requested split.
    */
    std::vector<int>& GetSplit(const ecvl::any& split = -1);

    /** @brief Set the current split.
    @param[in] split index, name or ecvl::SplitType representing the split to set.
    */
    virtual void SetSplit(const ecvl::any& split);

    /** @brief Dump the Dataset into a YAML file following the DeepHealth Dataset Format.

    The YAML file is saved into the dataset root directory.
    Samples paths are relative to the dataset root directory.

    @param[in] file_path Where to save the YAML file.
    */
    void Dump(const filesystem::path& file_path);

    /** @brief Retrieve the list of all samples locations in the dataset file.

    A single Sample can have multiple locations (e.g., if they are different acquisitions of the same image).

    @return vector containing all the samples locations.
    */
    std::vector<std::vector<filesystem::path>> GetLocations();

    // RegEx which matchs URLs
    static const std::regex url_regex_;
};
} // namespace ecvl

#endif // ECVL_DATASET_PARSER_H_