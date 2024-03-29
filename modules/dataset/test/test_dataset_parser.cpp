/*
* ECVL - European Computer Vision Library
* Version: 1.0.3
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include "ecvl/core.h"
#include "ecvl/dataset_parser.h"
#ifdef ECVL_WITH_EXAMPLES
#include "dataset_path.h"
#endif

#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace ecvl;

#ifdef ECVL_WITH_EXAMPLES
TEST(DatasetParser, LoadExistingDataset)
{
    Dataset d(CMAKE_CURRENT_SOURCE_DIR "/examples/data/mnist/mnist_reduced.yml");
    EXPECT_EQ(d.name_, "MNIST");
    EXPECT_EQ(d.classes_.size(), 10);
    EXPECT_THAT(d.classes_, testing::ElementsAre("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"));
    EXPECT_EQ(d.samples_.size(), 1000);
}
#endif

TEST(DatasetParser, LoadNonExistingDataset)
{
    EXPECT_THROW(Dataset d("idontexist"), std::runtime_error);
}

TEST(DatasetParser, LoadNonExistingOrBadImage)
{
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
