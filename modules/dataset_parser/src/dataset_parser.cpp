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

Image Sample::LoadImage(ColorType ctype) const
{
    bool status;
    Image img;

    status = ImRead(location_, img);
    if (!status) {
        // Image not correctly loaded, it is a URL!
        // TODO: Use libcurl instead of system call
        path image_filename = location_.filename();
        string cmd = "curl -s -o " + image_filename.string() + " " + location_.string();
        if (system(cmd.c_str()) != 0) {
            // Error, not a path nor a URL
            cout << ECVL_WARNING_MSG "Cannot load Sample '" + location_.string() + "', wrong path.\n";
        }
        else {
            ImRead(image_filename, img);
        }
    }

    if (img.colortype_ != ctype) {
        ChangeColorSpace(img, img, ctype);
    }
    return img;
}

void Dataset::DecodeImages(const YAML::Node& node, const path& root_path)
{
    // Allocate memory for the images
    this->samples_.resize(node.size());
    int counter = -1;
    // RegEx which matchs URLs
    std::regex r{ R"((http(s)?://.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*))" };

    for (auto& n : node) {
        // iterate over images
        auto& sample = this->samples_[++counter];
        if (n.IsScalar()) {
            // locations is provided as scalar without label or values
            sample.location_ = n.as<string>();
        }
        else {
            sample.location_ = n["location"].as<string>();

            if (n["label"].IsDefined()) {
                sample.label_ = vector<int>();
                if (n["label"].IsSequence()) {
                    for (int i = 0; i < n["label"].size(); ++i) {
                        FindLabel(sample, n["label"][i]);
                    }
                }
                else {
                    // label is a single value
                    FindLabel(sample, n["label"]);
                }
            }

            if (n["values"].IsDefined()) {
                sample.values_.emplace(map<int, string>());
                if (n["values"].IsSequence()) {
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
        }
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