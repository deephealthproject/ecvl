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