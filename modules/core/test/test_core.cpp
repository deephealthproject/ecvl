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

using namespace ecvl;

TEST(Core, CreateEmptyImage)
{
    Image img;
    EXPECT_EQ(img.dims_.size(), 0);
    EXPECT_EQ(img.strides_.size(), 0);
    EXPECT_EQ(img.data_, nullptr);
}

TEST(Core, CreateImageWithFiveDims)
{
    Image img({ 1, 2, 3, 4, 5 }, DataType::uint8, "xyzoo", ColorType::none);
    EXPECT_EQ(img.dims_.size(), 5);
    int sdims = vsize(img.dims_);
    for (int i = 0; i < sdims; i++) {
        EXPECT_EQ(img.dims_[i], i + 1);
    }
    EXPECT_EQ(img.strides_.size(), 5);
}

TEST(ArithmeticNeg, WorksWithInt8)
{
    Image x({ 5, 4, 3 }, DataType::int8, "xyc", ColorType::RGB);
    View<DataType::int8> y(x);
    y({ 0,0,0 }) = 15; y({ 1,0,0 }) = 16; y({ 2,0,0 }) = 17; y({ 3,0,0 }) = 18; y({ 4,0,0 }) = 19;
    y({ 0,1,0 }) = 25; y({ 1,1,0 }) = 26; y({ 2,1,0 }) = 27; y({ 3,1,0 }) = 28; y({ 4,1,0 }) = 29;
    y({ 0,2,0 }) = 35; y({ 1,2,0 }) = 36; y({ 2,2,0 }) = 37; y({ 3,2,0 }) = -128; y({ 4,2,0 }) = 39;
    y({ 0,3,0 }) = 45; y({ 1,3,0 }) = 46; y({ 2,3,0 }) = 47; y({ 3,3,0 }) = 48; y({ 4,3,0 }) = 49;

    y({ 0,0,1 }) = 17; y({ 1,0,1 }) = 16; y({ 2,0,1 }) = 10; y({ 3,0,1 }) = 17; y({ 4,0,1 }) = 19;
    y({ 0,1,1 }) = 27; y({ 1,1,1 }) = 26; y({ 2,1,1 }) = 20; y({ 3,1,1 }) = 27; y({ 4,1,1 }) = 29;
    y({ 0,2,1 }) = 37; y({ 1,2,1 }) = 36; y({ 2,2,1 }) = 30; y({ 3,2,1 }) = 37; y({ 4,2,1 }) = -127;
    y({ 0,3,1 }) = 47; y({ 1,3,1 }) = 46; y({ 2,3,1 }) = 40; y({ 3,3,1 }) = 47; y({ 4,3,1 }) = 49;

    y({ 0,0,2 }) = 15; y({ 1,0,2 }) = 17; y({ 2,0,2 }) = 17; y({ 3,0,2 }) = 18; y({ 4,0,2 }) = 17;
    y({ 0,1,2 }) = 25; y({ 1,1,2 }) = 27; y({ 2,1,2 }) = 27; y({ 3,1,2 }) = 28; y({ 4,1,2 }) = 27;
    y({ 0,2,2 }) = 35; y({ 1,2,2 }) = 37; y({ 2,2,2 }) = 37; y({ 3,2,2 }) = 38; y({ 4,2,2 }) = 37;
    y({ 0,3,2 }) = 45; y({ 1,3,2 }) = 47; y({ 2,3,2 }) = 47; y({ 3,3,2 }) = 48; y({ 4,3,2 }) = 47;

    Neg(x);

    EXPECT_EQ(y({ 1,2,0 }), -36);
    EXPECT_EQ(y({ 3,3,2 }), -48);
    EXPECT_EQ(y({ 4,2,1 }), 127);
    EXPECT_EQ(y({ 3,2,0 }), -128);
}

TEST(RearrangeChannels, WorksWithVolumeInt16RGB)
{
    Image img({ 3, 4, 3, 2 }, DataType::int16, "cxyz", ColorType::RGB);
    View<DataType::int16> view(img);
    auto it = view.Begin();
    for (int i = 0; i < 24 * 3; i++) {
        *it = i;
        ++it;
    }
    Image img2;
    RearrangeChannels(img, img2, "xyzc");
    View<DataType::int16> view2(img2);

    EXPECT_EQ(view2({ 2, 0, 1, 0 }), 42);
    EXPECT_EQ(view2({ 3, 1, 1, 2 }), 59);
    EXPECT_EQ(view2({ 0, 2, 0, 1 }), 25);
    EXPECT_EQ(view2({ 1, 2, 0, 1 }), 28);
}