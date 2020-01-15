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

#ifndef ECVL_DATASET_PARSER_H_
#define ECVL_DATASET_PARSER_H_

#include "ecvl/core.h"

#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <vector>

#include "yaml-cpp/yaml.h"

namespace ecvl {
/** @brief Sample image in a dataset.

This class provides the information to describe a dataset sample.
`label_` and `label_path_` are mutually exclusive.
@anchor Sample
*/
class Sample {
public:
    std::filesystem::path location_; /**< @brief Absolute path of the sample. */
    std::optional<std::vector<int>> label_; /**< @brief Vector of sample labels. */
    std::optional<std::filesystem::path> label_path_; /**< @brief Absolute path of sample ground truth. */
    std::optional<std::map<int, std::string>> values_; /**< @brief Map (`map<feature-index,feature-value>`) which stores the features of a sample. */

    /** @brief Return an Image of the dataset.

    The LoadImage() function opens the sample image, from `location_` or `label_path_` depending on `is_gt` parameter.

    @param[in] ctype ecvl::ColorType of the returned Image.
    @param[in] is_gt Whether to load the sample image or its ground truth.

    @return Image containing the loaded sample.
    */
    ecvl::Image LoadImage(ecvl::ColorType ctype = ecvl::ColorType::BGR, const bool& is_gt = false) const;
};

/** @brief Splits of a dataset.

This class provides the splits a dataset can have: training, validation, and test.

@anchor Split
*/
class Split {
public:
    std::vector<int> training_;   /**< @brief Vector containing samples of training split. */
    std::vector<int> validation_; /**< @brief Vector containing samples of validation split. */
    std::vector<int> test_;       /**< @brief Vector containing samples of test split. */
};

/** @brief DeepHealth Dataset.

This class implements the DeepHealth Dataset Format (https://github.com/deephealthproject/ecvl/wiki/DeepHealth-Toolkit-Dataset-Format).

@anchor Dataset
*/
class Dataset {
public:
    std::string name_ = "DeepHealth dataset"; /**< @brief Name of the Dataset. */
    std::string description_ = "This is the DeepHealth example dataset!"; /**< @brief Description of the Dataset. */
    std::vector<std::string> classes_; /**< @brief Vector with all the classes available in the Dataset. */
    std::vector<std::string> features_; /**< @brief Vector with all the features available in the Dataset. */
    std::vector<Sample> samples_; /**< @brief Vector containing all the Dataset samples. See @ref Sample. */
    Split split_; /**< @brief Splits of the Dataset. See @ref Split. */

    Dataset() {}

    /**
    @param[in] filename Path to the Dataset file.
    */
    Dataset(const std::filesystem::path& filename);

private:
    std::map<std::string, int> features_map_;
    void DecodeImages(const YAML::Node& node, const std::filesystem::path& root_path);
    void FindLabel(Sample& sample, const YAML::Node& n);
};
} // namespace ecvl

/** @cond HIDDEN_SECTIONS */
namespace YAML {
/**
    Enable YAML decoding of Split.
    Hidden from docs.
*/
template<>
struct convert<ecvl::Split> {
    /*static Node encode(const ecvl::Split& rhs)
    {
        Node node;
        node.push_back(rhs.x);
        return node;
    }*/

    static bool decode(const YAML::Node& node, ecvl::Split& rhs)
    {
        if (node["training"].IsDefined()) {
            rhs.training_ = node["training"].as<std::vector<int>>();
        }
        if (node["validation"].IsDefined()) {
            rhs.validation_ = node["validation"].as<std::vector<int>>();
        }
        if (node["test"].IsDefined()) {
            rhs.test_ = node["test"].as<std::vector<int>>();
        }
        return true;
    }
};
} // namespace YAML
/** @endcond */

#endif // ECVL_DATASET_PARSER_H_