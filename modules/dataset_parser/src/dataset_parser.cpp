#include "ecvl/dataset_parser.h"

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

void Dataset::LoadImage(Sample& sample, const path& root_path)
{
    bool status;
    if (sample.location_.is_absolute()) {
        // Absolute path
        status = ImRead(sample.location_, sample.image_);
    }
    else {
        // Relative path
        path location = root_path / sample.location_;
        status = ImRead(location, sample.image_);
    }

    if (!status) {
        // Image not correctly loaded, it is a URL!
        // TODO: Use libcurl instead of system call
        path image_filename = sample.location_.filename();
        string cmd = "curl -s -o " + image_filename.string() + " " + sample.location_.string();
        if (system(cmd.c_str()) != 0) {
            // Error, not a path nor a URL
            cout << ECVL_WARNING_MSG "Cannot load Sample '" + sample.location_.string() + "', wrong path.\n";
        }
        else {
            ImRead(image_filename, sample.image_);
        }
    }
}

void Dataset::DecodeImages(const YAML::Node& node, const path& root_path)
{
    // Allocate memory for the images
    this->images_.resize(node.size());
    int counter = -1;

    for (auto& n : node) {
        // iterate over images
        auto& img = this->images_[++counter];
        if (n.IsScalar()) {
            // locations is provided as scalar without label or values
            img.location_ = n.as<string>();
        }
        else {
            img.location_ = n["location"].as<string>();

            if (n["label"].IsDefined()) {
                img.label_ = vector<int>();
                if (n["label"].IsSequence()) {
                    for (int i = 0; i < n["label"].size(); ++i) {
                        FindLabel(img, n["label"][i]);
                    }
                }
                else {
                    // label is a single value
                    FindLabel(img, n["label"]);
                }
            }

            if (n["values"].IsDefined()) {
                img.values_.emplace(map<int, string>());
                if (n["values"].IsSequence()) {
                    for (int i = 0; i < n["values"].size(); ++i) {
                        if (!n["values"][i].IsNull()) {
                            img.values_.value()[i] = n["values"][i].as<string>();
                        }
                    }
                }
                else {
                    // values is a dictionary
                    for (YAML::const_iterator it = n["values"].begin(); it != n["values"].end(); ++it) {
                        img.values_.value()[this->features_map_[it->first.as<string>()]] = it->second.as<string>();
                    }
                }
            }
        }
        LoadImage(img, root_path);
    }
}

Dataset::Dataset(const path& filename)
{
    path abs_filename = absolute(filename);
    YAML::Node config;
    try {
        config = YAML::LoadFile(abs_filename.string());
    }
    catch (Exception& e) {
        cout << "ERROR: Unable to read dataset file '" << abs_filename << "'.\n";
        cout << "MSG: " << e.what();
        exit(EXIT_FAILURE);
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

    DecodeImages(config["images"], abs_filename.parent_path());
    if (config["split"].IsDefined()) {
        this->split_ = config["split"].as<Split>();
    }
}