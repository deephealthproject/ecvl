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

#include <gtest/gtest.h>
#include "ecvl/core.h"
#include "ecvl/dataset_parser.h"
#include <fstream>

using namespace ecvl;

TEST(DatasetParser, LoadExistingDataset) {
    Dataset d("../examples/data/mnist/mnist.yml");
    EXPECT_EQ(d.name_, "MNIST");
    std::vector<std::string> classes{ "0","1","2","3","4","5","6","7","8","9" };
    EXPECT_EQ(d.classes_.size(), 10);
    for (size_t i = 0; i < d.classes_.size(); ++i) {
        EXPECT_EQ(classes[i], d.classes_[i]);
    }
    EXPECT_EQ(d.samples_.size(), 70000);
}

TEST(DatasetParser, LoadNonExistingDataset) {
    EXPECT_THROW(Dataset d("idontexist"), std::runtime_error);
}

TEST(DatasetParser, LoadNonExistingOrBadImage) {
    {
        std::ofstream os("hello.yml");
        os << "dataset: test\n"
            "images:\n"
            "    - location: idontexist\n"
            "      label: hello.png\n";
    }
    {
        std::ofstream os("hello.png");
        os << "this is not a valid image";
    }
    Dataset d("hello.yml");
    EXPECT_THROW(d.samples_.front().LoadImage(), std::runtime_error);
    EXPECT_THROW(d.samples_.front().LoadImage(ColorType::GRAY, true), std::runtime_error);
    remove("hello.yml");
    remove("hello.png");
}
