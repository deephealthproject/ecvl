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
#include "ecvl/core/filesystem.h"
#include "ecvl/core/optional.h"

#include <iostream>
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
/** @brief Sample image in a dataset.

This class provides the information to describe a dataset sample.
`label_` and `label_path_` are mutually exclusive.
@anchor Sample
*/
class Sample
{
public:
    std::vector<filesystem::path> location_; /**< @brief Absolute path of the sample. */
    optional<std::vector<int>> label_; /**< @brief Vector of sample labels. */
    optional<filesystem::path> label_path_; /**< @brief Absolute path of sample ground truth. */
    optional<std::map<int, std::string>> values_; /**< @brief Map (`map<feature-index,feature-value>`) which stores the features of a sample. */
    std::vector<int> size_; /**< @brief Original x and y dimensions of the sample */

    /** @brief Return an Image of the dataset.

    The LoadImage() function opens the sample image, from `location_` or `label_path_` depending on `is_gt` parameter.

    @param[in] ctype ecvl::ColorType of the returned Image.
    @param[in] is_gt Whether to load the sample image or its ground truth.

    @return Image containing the loaded sample.
    */
    ecvl::Image LoadImage(ecvl::ColorType ctype = ecvl::ColorType::BGR, const bool& is_gt = false);
};

/** @brief DeepHealth Dataset.

This class implements the DeepHealth Dataset Format (https://github.com/deephealthproject/ecvl/wiki/DeepHealth-Toolkit-Dataset-Format).

@anchor Dataset
*/
class Dataset
{
public:
    std::string name_ = "DeepHealth dataset"; /**< @brief Name of the Dataset. */
    std::string description_ = "This is the DeepHealth example dataset!"; /**< @brief Description of the Dataset. */
    std::vector<std::string> classes_; /**< @brief Vector with all the classes available in the Dataset. */
    std::vector<std::string> features_; /**< @brief Vector with all the features available in the Dataset. */
    std::vector<Sample> samples_; /**< @brief Vector containing all the Dataset samples. See @ref Sample. */
    std::vector<std::pair<std::string, std::vector<int>>> split_; /**< @brief Splits of the Dataset. */
    int current_split_ = -1; /**< @brief Current split from which images are loaded. */

    Dataset() {}

    /**
    @param[in] filename Path to the Dataset file.
    @param[in] verify Whether to log the non-existence of a dataset sample location or not.
    */
    Dataset(const filesystem::path& filename, bool verify = false);

    /** @brief Returns the image indexes of the current split.
    @return vector of image indexes of the split in use.
    */
    std::vector<int>& GetSplit();

    /** @brief Returns the image indexes of the requested split.
    @param[in] split ecvl::SplitType representing the split to get ("training", "validation", or "test").
    @return vector of image indexes of the requested split.
    */
    std::vector<int>& GetSplit(const SplitType& split);

    /** @brief Returns the image indexes of the requested split.
    @param[in] split string representing the split to get.
    @return vector of image indexes of the requested split.
    */
    std::vector<int>& GetSplit(const std::string& split);

    /** @brief Returns the image indexes of the requested split.
    @param[in] split int representing the index of the split to get.
    @return vector of image indexes of the requested split.
    */
    std::vector<int>& GetSplit(const int& split);

    /** @brief Set the current split.
    @param[in] split ecvl::SplitType representing the split to set ("training", "validation", or "test").
    */
    void SetSplit(const SplitType& split);

    /** @brief Set the current split.
    @param[in] split string representing the split to set.
    */
    void SetSplit(const std::string& split);

    /** @brief Set the current split.
    @param[in] split int representing the index of the split to set.
    */
    void SetSplit(const int& split);

    /** @brief Dump the Dataset into a YAML file following the DeepHealth Dataset Format.

    The YAML file is saved into the dataset root directory.
    Samples paths are relative to the dataset root directory.

    @param[in] file_path Where to save the YAML file.
    */
    void Dump(const filesystem::path& file_path);

    // RegEx which matchs URLs
    static const std::regex url_regex_;

private:
    std::map<std::string, int> features_map_;
    void DecodeImages(const YAML::Node& node, const filesystem::path& root_path, bool verify);
    void FindLabel(Sample& sample, const YAML::Node& n);
};

/** @brief Convert @ref SplitType in string.

Useful for backward compatibility.

@param[in] split SplitType to convert

@return string that represent the provided SplitType
*/
const std::string SplitTypeToString(const SplitType& split);
} // namespace ecvl

#endif // ECVL_DATASET_PARSER_H_