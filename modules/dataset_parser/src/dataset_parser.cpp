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

#include "ecvl/dataset_parser.h"
#include <regex>

using namespace std;
using namespace std::filesystem;
using namespace YAML;
using namespace ecvl;

void Dataset::FindLabel(Sample& sample, const YAML::Node& n)
{
    vector<string>::iterator it;

    try {
        // it is a int
        sample.label_->push_back(n.as<int>());
    }
    catch (const BadConversion) {
        // it's a string!
        string s = n.as<string>();
        it = std::find(this->classes_.begin(), this->classes_.end(), s);
        if (it != this->classes_.end()) {
            int index = static_cast<int>(distance(this->classes_.begin(), it));
            sample.label_->push_back(index);
        }
        else {
            // TODO label as path?
            sample.label_path_ = n.as<string>();
            sample.label_.reset();
        }
    }
}

Image Sample::LoadImage(ColorType ctype, const bool& is_gt) const
{
    bool status;
    Image img;
    static std::regex r{ R"(https?://.*)" };

    auto location = is_gt ? label_path_.value() : location_;
    if (std::regex_match(location.string(), r)) {
        // TODO: Use libcurl instead of system call
        path image_filename = location.filename();
        string cmd = "curl -s -o " + image_filename.string() + " " + location.string();
        if (system(cmd.c_str()) != 0) {
            // Failed to download image
            cerr << ECVL_ERROR_MSG "Cannot download image '" + location.string() + "'.\n";
            ECVL_ERROR_CANNOT_LOAD_FROM_URL
        }
        else {
            location = image_filename;
        }
    }

    if (!filesystem::exists(location)) {
        cerr << ECVL_ERROR_MSG "image " << location << " does not exist" << endl;
        ECVL_ERROR_FILE_DOES_NOT_EXIST
    }

    status = ImRead(location, img);

    if (!status) {
        // Image not correctly loaded
        cerr << ECVL_ERROR_MSG "Cannot load image '" + location.string() + "'.\n";
        ECVL_ERROR_CANNOT_LOAD_IMAGE
    }

    if (img.colortype_ != ctype) {
        ChangeColorSpace(img, img, ctype);
    }
    return img;
}

void Dataset::DecodeImages(const YAML::Node& node, const path& root_path, bool verify)
{
    // Allocate memory for the images
    this->samples_.resize(node.size());
    int counter = -1;
    // RegEx which matchs URLs
    std::regex r{ R"(https?://.*)" };

    for (auto& n : node) {
        // iterate over images
        auto& sample = this->samples_[++counter];
        if (n.IsScalar()) {
            // locations is provided as scalar without label or values
            sample.location_ = n.as<string>();
        }
        else {
            sample.location_ = n["location"].as<string>();

            // Load labels
            if (n["label"].IsDefined()) {
                sample.label_ = vector<int>();
                if (n["label"].IsSequence()) {
                    // values is a list
                    for (int i = 0; i < n["label"].size(); ++i) {
                        FindLabel(sample, n["label"][i]);
                    }
                }
                else {
                    // label is a single value
                    FindLabel(sample, n["label"]);
                }
            }

            // Load features
            if (n["values"].IsDefined()) {
                sample.values_.emplace(map<int, string>());
                if (n["values"].IsSequence()) {
                    // values is a list
                    for (int i = 0; i < n["values"].size(); ++i) {
                        if (!n["values"][i].IsNull()) {
                            sample.values_.value()[i] = n["values"][i].as<string>();
                        }
                    }
                }
                else {
                    // values is a dictionary
                    for (YAML::const_iterator it = n["values"].begin(); it != n["values"].end(); ++it) {
                        sample.values_.value()[this->features_map_[it->first.as<string>()]] = it->second.as<string>();
                    }
                }
            }
        }
        if (sample.location_.is_relative() && !std::regex_match(sample.location_.string(), r)) {
            // Convert relative path to absolute
            sample.location_ = root_path / sample.location_;
            if (sample.label_path_.has_value()) {
                sample.label_path_ = root_path / sample.label_path_.value();
            }
        }
        if (verify) {
            if (!filesystem::exists(sample.location_)) {
                cerr << ECVL_WARNING_MSG "sample file " << sample.location_ << " does not exist" << endl;
            }
            if (sample.label_path_.has_value()) {
                if (!filesystem::exists(sample.label_path_.value())) {
                    cerr << ECVL_WARNING_MSG "label file " << sample.label_path_.value() << " does not exist" << endl;
                }
            }
        }
    }
}

Dataset::Dataset(const path& filename, bool verify)
{
    path abs_filename = absolute(filename);

    if (!filesystem::exists(abs_filename)) {
        cerr << ECVL_ERROR_MSG "dataset file " << filename << " does not exist" << endl;
        ECVL_ERROR_FILE_DOES_NOT_EXIST
    }

    YAML::Node config;
    try {
        config = YAML::LoadFile(abs_filename.string());
    }
    catch (const YAML::BadFile& e) {
        cerr << ECVL_ERROR_MSG "parse of dataset file " << filename << " failed." << endl;
        cerr << "MSG: " << e.what();
        ECVL_ERROR_NOT_REACHABLE_CODE
    }

    if (config["name"].IsDefined()) {
        this->name_ = config["name"].as<string>();
    }
    if (config["description"].IsDefined()) {
        this->description_ = config["description"].as<string>();
    }
    if (config["classes"].IsDefined()) {
        this->classes_ = config["classes"].as<vector<string>>();
    }
    if (config["features"].IsDefined()) {
        this->features_ = config["features"].as<vector<string>>();
        for (int i = 0; i < this->features_.size(); ++i) {
            this->features_map_[this->features_[i]] = i;
        }
    }

    DecodeImages(config["images"], abs_filename.parent_path(), verify);
    if (config["split"].IsDefined()) {
        this->split_ = config["split"].as<Split>();
    }
}