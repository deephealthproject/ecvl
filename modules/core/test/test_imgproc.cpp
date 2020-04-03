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

TEST(Imgproc, Thresholding)
{
    Image x({ 2, 2, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
    View<DataType::uint8> x_v(x);

    x_v({ 0,0,0 }) = 250;
    x_v({ 1,0,0 }) = 16;
    x_v({ 0,1,0 }) = 15;
    x_v({ 1,1,0 }) = 200;

    Image o;
    Threshold(x, o, 127, 255, ThresholdingType::BINARY);
    View<DataType::uint8> o_v(o);

    EXPECT_EQ(o_v({ 0,0,0 }), 255);
    EXPECT_EQ(o_v({ 1,0,0 }), 0);
    EXPECT_EQ(o_v({ 0,1,0 }), 0);
    EXPECT_EQ(o_v({ 1,1,0 }), 255);

    Threshold(x, o, 127, 255, ThresholdingType::BINARY_INV);
    o_v = o;

    EXPECT_EQ(o_v({ 0,0,0 }), 0);
    EXPECT_EQ(o_v({ 1,0,0 }), 255);
    EXPECT_EQ(o_v({ 0,1,0 }), 255);
    EXPECT_EQ(o_v({ 1,1,0 }), 0);
}

TEST(Imgproc, Mirroring)
{
    Image x({ 2, 2, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
    Image y;

    *x.Ptr({ 0,0,0 }) = 0; *x.Ptr({ 1,0,0 }) = 1;
    *x.Ptr({ 0,1,0 }) = 1; *x.Ptr({ 1,1,0 }) = 0;
    Mirror2D(x, y);

    EXPECT_EQ(*y.Ptr({ 0,0,0 }), 1); EXPECT_EQ(*y.Ptr({ 1,0,0 }), 0);
    EXPECT_EQ(*y.Ptr({ 0,1,0 }), 0); EXPECT_EQ(*y.Ptr({ 1,1,0 }), 1);

    // Test 1x1 Image
    x = Image({ 1, 1, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
    *x.Ptr({ 0,0,0 }) = 0;
    Mirror2D(x, y);

    // Test 3x3 Image
    x = Image({ 3, 3, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
    *x.Ptr({ 0,0,0 }) = 1; *x.Ptr({ 1,0,0 }) = 2; *x.Ptr({ 2,0,0 }) = 3;
    *x.Ptr({ 0,1,0 }) = 1; *x.Ptr({ 1,1,0 }) = 2; *x.Ptr({ 2,1,0 }) = 3;
    *x.Ptr({ 0,2,0 }) = 1; *x.Ptr({ 1,2,0 }) = 2; *x.Ptr({ 2,2,0 }) = 3;
    Mirror2D(x, y);
    EXPECT_EQ(*y.Ptr({ 0,0,0 }), 3); EXPECT_EQ(*y.Ptr({ 1,0,0 }), 2); EXPECT_EQ(*y.Ptr({ 2,0,0 }), 1);
    EXPECT_EQ(*y.Ptr({ 0,1,0 }), 3); EXPECT_EQ(*y.Ptr({ 1,1,0 }), 2); EXPECT_EQ(*y.Ptr({ 2,1,0 }), 1);
    EXPECT_EQ(*y.Ptr({ 0,2,0 }), 3); EXPECT_EQ(*y.Ptr({ 1,2,0 }), 2); EXPECT_EQ(*y.Ptr({ 2,2,0 }), 1);
}