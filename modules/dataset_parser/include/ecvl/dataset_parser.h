#ifndef ECVL_DATASET_PARSER_H_
#define ECVL_DATASET_PARSER_H_

#include "ecvl/core.h"

#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <vector>

#include "yaml-cpp/yaml.h"

class Sample {
public:
    std::filesystem::path location_;
    std::optional<std::vector<int>> label_;
    std::optional<std::string> label_path_; // path to ground truth
    std::optional<std::map<int, std::string>> values_; //features

    ecvl::Image LoadImage(ecvl::ColorType ctype = ecvl::ColorType::BGR) const;
};

class Split {
public:
    std::vector<int> training_;
    std::vector<int> validation_;
    std::vector<int> test_;
};

class Dataset {
public:
    std::string name_ = "DeepHealth dataset";
    std::string description_ = "This is the DeepHealth example dataset!";
    std::vector<std::string> classes_;
    std::vector<std::string> features_;
    std::vector<Sample> samples_;
    Split split_;

    Dataset() {}
    Dataset(const std::filesystem::path& filename);

private:
    std::map<std::string, int> features_map_;
    void DecodeImages(const YAML::Node& node, const std::filesystem::path& root_path);
    void FindLabel(Sample& sample, const YAML::Node& n);
};

namespace YAML {
template<>
struct convert<Split> {
    /*static Node encode(const Split& rhs)
    {
        Node node;
        node.push_back(rhs.x);
        return node;
    }*/

    static bool decode(const YAML::Node& node, Split& rhs)
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
}

#endif // ECVL_DATASET_PARSER_H_