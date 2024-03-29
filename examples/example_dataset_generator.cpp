/*
* ECVL - European Computer Vision Library
* Version: 1.0.3
* copyright (c) 2021, UniversitÓ degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include "ecvl/core.h"
#include "ecvl/core/filesystem.h"
#include "ecvl/dataset_generator.h"

#include <iostream>

using namespace ecvl;
using namespace ecvl::filesystem;
using namespace std;

int main()
{
    path dateset_root_folder_classification("../examples/data/fake_classification_dataset");
    path dateset_root_folder_segmentation("../examples/data/fake_segmentation_dataset");
    Dataset d_classification, d_segmentation;

    // Classification dataset
    GenerateClassificationDataset c(dateset_root_folder_classification);
    d_classification = c.GetDataset();

    // Dump the Dataset on file
    d_classification.Dump(dateset_root_folder_classification / path(dateset_root_folder_classification.stem().string() + ".yml"));

    // Segmentation dataset
    path suffix = "_segmentation.png"; // Possible ground truth suffix or extension if different from images
    path gt_name = "black.png";        // Possible ground truth name for images that have the same ground truth
    GenerateSegmentationDataset s(dateset_root_folder_segmentation, suffix, gt_name);

    d_segmentation = s.GetDataset();

    // Do stuff with the Dataset, for example remove from training set all the images with a "black" ground truth
    vector<int> mask;
    vector<int> black;

    // auto& training = d_segmentation.GetSplit("training"); // Or
    auto& training = d_segmentation.GetSplit(SplitType::training);
    for (auto& sample_index : training) {
        if (d_segmentation.samples_[sample_index].label_path_.value().filename().compare("black.png") == 0) {
            black.emplace_back(sample_index);
        }
        else {
            mask.emplace_back(sample_index);
        }
    }

    training.clear();
    training.insert(training.end(), mask.begin(), mask.end());

    // Dump the Dataset on file
    d_segmentation.Dump(dateset_root_folder_segmentation / path(dateset_root_folder_segmentation.stem().string() + ".yml"));

    return EXIT_SUCCESS;
}