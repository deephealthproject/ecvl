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

#include "ecvl/dataset_generator.h"
#include <iostream>
#include <optional>

using namespace std;
using namespace std::filesystem;
using namespace ecvl;

int GenerateClassificationDataset::LoadSplitImages(const path& split)
{
    int n_samples = 0;
    int label_counter = -1;
    path label;
    path root_directory = dataset_root_directory_ / split;

    for (auto& p : recursive_directory_iterator(root_directory)) {
        if (is_directory(p.path())) {
            label = p.path().stem();
            ++label_counter;
        }
        else {
            if (p.path().parent_path().stem() == label) {
                Sample s;
                s.location_.emplace_back(root_directory.stem() / label / p.path().filename().generic_string());
                s.label_ = vector<int>{ label_counter };
                ++n_samples;
                d_.samples_.push_back(s);
            }
            else {
                cout << ECVL_WARNING_MSG "File '" + p.path().generic_string() + "' doesn't have an allowable label and it isn't added to the dataset file." << endl;
            }
        }
    }

    return n_samples;
}

int GenerateSegmentationDataset::LoadSplitImages(const path& split)
{
    int n_samples = 0;
    bool gt_exist = false;
    string images, ground_truth;
    path root_directory = dataset_root_directory_ / split;

    if (exists(root_directory / "images")) {
        images = "images";
        ground_truth = "ground_truth";
    }
    else {
        images = ground_truth = "";
    }
    path gt_img;

    for (auto& p : directory_iterator(root_directory / images)) {
        gt_exist = false;
        if (!is_directory(p.path()) && p.path().extension() != ".yml") {
            string img = (split / images / p.path().filename()).generic_string();

            Sample s;
            s.location_.emplace_back(img);

            if (suffix_.empty() && gt_name_.empty()) {
                gt_img = root_directory / ground_truth / p.path().filename();
                if (exists(gt_img) && !ground_truth.empty()) {
                    s.label_path_ = (split / ground_truth / gt_img.filename()).generic_string();
                    gt_exist = true;
                }
                else if (split != "test") { // test may not have ground truth images
                    if (splits_.empty()) {
                        if (!ground_truth.empty()) {
                            cout << ECVL_WARNING_MSG "File '" + gt_img.generic_string() + "' doesn't exist and it isn't added to the dataset file." << endl;
                        }
                    }
                    else {
                        cout << ECVL_WARNING_MSG "File '" + img + "' doesn't have an existing ground truth and it isn't added to the dataset file." << endl;
                    }
                }
            }
            else {
                if ((!suffix_.empty() && img.find(suffix_.string()) != string::npos) || 
                    (!gt_name_.empty() && img.find(gt_name_.string()) != string::npos)) {
                    // suffix found, it is a gt -> skip
                    continue;
                }

                gt_img = root_directory / ground_truth / path(p.path().stem().string() + suffix_.string());

                if (exists(gt_img)) {
                    s.label_path_ = (split / ground_truth / gt_img.filename()).generic_string();
                    gt_exist = true;
                }
                else if (exists(root_directory / ground_truth / gt_name_) && !gt_name_.empty() ) {
                    s.label_path_ = (split / ground_truth / gt_name_).generic_string();
                    gt_exist = true;
                }
                else if (split != "test") {
                    if (splits_.empty()) {
                        cout << ECVL_WARNING_MSG "File '" + gt_img.generic_string() + "' doesn't exist and it isn't added to the dataset file." << endl;
                    }
                    else {
                        cout << ECVL_WARNING_MSG "File '" + img + "' doesn't have an existing ground truth and it isn't added to the dataset file." << endl;
                    }
                }
            }

            // for training and validation splits it is mandatory to have ground truth images
            if (split == "test" || gt_exist || splits_.empty()) {
                d_.samples_.push_back(s);
                ++n_samples;
            }
        }
    }

    return n_samples;
}

void GenerateDataset::LoadImagesAndSplits()
{
    if (splits_.empty()) {
        // load locations and labels
        this->LoadSplitImages("");
    }
    else {
        int img_index = 0;

        // load locations and labels and get the number of samples for each split
        for (int i = 0; i < splits_.size(); ++i) {
            num_samples_[i] = this->LoadSplitImages(splits_[i]);
        }

        // load indexes of images for each split
        for (int i = 0; i < splits_.size(); ++i) {
            if (splits_[i] == "training") {
                d_.split_.training_.resize(num_samples_[i]);
                iota(d_.split_.training_.begin(), d_.split_.training_.end(), img_index);
            }
            else if (splits_[i] == "validation") {
                d_.split_.validation_.resize(num_samples_[i]);
                iota(d_.split_.validation_.begin(), d_.split_.validation_.end(), img_index);
            }
            else if (splits_[i] == "test") {
                d_.split_.test_.resize(num_samples_[i]);
                iota(d_.split_.test_.begin(), d_.split_.test_.end(), img_index);
            }
            img_index += num_samples_[i];
        }
    }
}