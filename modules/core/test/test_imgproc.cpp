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
    Image o;

    x_v({ 0,0,0 }) = 250; x_v({ 1,0,0 }) = 16;
    x_v({ 0,1,0 }) = 15; x_v({ 1,1,0 }) = 200;

    Threshold(x, o, 127, 255, ThresholdingType::BINARY);
    View<DataType::uint8> o_v(o);

    EXPECT_EQ(o_v({ 0,0,0 }), 255); EXPECT_EQ(o_v({ 1,0,0 }), 0);
    EXPECT_EQ(o_v({ 0,1,0 }), 0); EXPECT_EQ(o_v({ 1,1,0 }), 255);

    Threshold(x, o, 127, 255, ThresholdingType::BINARY_INV);
    o_v = o;

    EXPECT_EQ(o_v({ 0,0,0 }), 0); EXPECT_EQ(o_v({ 1,0,0 }), 255);
    EXPECT_EQ(o_v({ 0,1,0 }), 255); EXPECT_EQ(o_v({ 1,1,0 }), 0);
}

TEST(Imgproc, Mirroring)
{
    Image x({ 2, 2, 1 }, DataType::float32, "xyc", ColorType::GRAY);
    Image y;

    {
        View<DataType::float32> x_v(x);
        x_v({ 0,0,0 }) = 0; x_v({ 1,0,0 }) = 1;
        x_v({ 0,1,0 }) = 1; x_v({ 1,1,0 }) = 0;
        Mirror2D(x, y);

        View<DataType::float32> y_v(y);
        EXPECT_FLOAT_EQ(y_v({ 0,0,0 }), 1); EXPECT_FLOAT_EQ(y_v({ 1,0,0 }), 0);
        EXPECT_FLOAT_EQ(y_v({ 0,1,0 }), 0); EXPECT_FLOAT_EQ(y_v({ 1,1,0 }), 1);
    }
    {
        // Test 1x1 Image
        x = Image({ 1, 1, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
        View<DataType::uint8> x_v(x);

        x_v({ 0,0,0 }) = 1;
        Mirror2D(x, y);
        View<DataType::uint8> y_v(y);
        EXPECT_EQ(y_v({ 0,0,0 }), 1);

        // Test 3x3 Image
        x = Image({ 3, 3, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
        x_v = x;

        x_v({ 0,0,0 }) = 1; x_v({ 1,0,0 }) = 2; x_v({ 2,0,0 }) = 3;
        x_v({ 0,1,0 }) = 1; x_v({ 1,1,0 }) = 2; x_v({ 2,1,0 }) = 3;
        x_v({ 0,2,0 }) = 1; x_v({ 1,2,0 }) = 2; x_v({ 2,2,0 }) = 3;
        Mirror2D(x, y);
        y_v = y;
        EXPECT_EQ(y_v({ 0,0,0 }), 3); EXPECT_EQ(y_v({ 1,0,0 }), 2); EXPECT_EQ(y_v({ 2,0,0 }), 1);
        EXPECT_EQ(y_v({ 0,1,0 }), 3); EXPECT_EQ(y_v({ 1,1,0 }), 2); EXPECT_EQ(y_v({ 2,1,0 }), 1);
        EXPECT_EQ(y_v({ 0,2,0 }), 3); EXPECT_EQ(y_v({ 1,2,0 }), 2); EXPECT_EQ(y_v({ 2,2,0 }), 1);
    }
}

TEST(Imgproc, Flipping)
{
    Image x({ 2, 2, 1 }, DataType::float32, "xyc", ColorType::GRAY);
    Image y;

    {
        View<DataType::float32> x_v(x);
        x_v({ 0,0,0 }) = 0; x_v({ 1,0,0 }) = 1;
        x_v({ 0,1,0 }) = 1; x_v({ 1,1,0 }) = 0;
        Flip2D(x, y);

        View<DataType::float32> y_v(y);
        EXPECT_FLOAT_EQ(y_v({ 0,0,0 }), 1); EXPECT_FLOAT_EQ(y_v({ 1,0,0 }), 0);
        EXPECT_FLOAT_EQ(y_v({ 0,1,0 }), 0); EXPECT_FLOAT_EQ(y_v({ 1,1,0 }), 1);
    }
    {
        // Test 1x1 Image
        x = Image({ 1, 1, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
        View<DataType::uint8> x_v(x);

        x_v({ 0,0,0 }) = 1;
        Flip2D(x, y);
        View<DataType::uint8> y_v(y);
        EXPECT_EQ(y_v({ 0,0,0 }), 1);

        // Test 3x3 Image
        x = Image({ 3, 3, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
        x_v = x;
        x_v({ 0,0,0 }) = 1; x_v({ 1,0,0 }) = 1; x_v({ 2,0,0 }) = 1;
        x_v({ 0,1,0 }) = 2; x_v({ 1,1,0 }) = 2; x_v({ 2,1,0 }) = 2;
        x_v({ 0,2,0 }) = 3; x_v({ 1,2,0 }) = 3; x_v({ 2,2,0 }) = 3;
        Flip2D(x, y);
        y_v = y;
        EXPECT_EQ(y_v({ 0,0,0 }), 3); EXPECT_EQ(y_v({ 1,0,0 }), 3); EXPECT_EQ(y_v({ 2,0,0 }), 3);
        EXPECT_EQ(y_v({ 0,1,0 }), 2); EXPECT_EQ(y_v({ 1,1,0 }), 2); EXPECT_EQ(y_v({ 2,1,0 }), 2);
        EXPECT_EQ(y_v({ 0,2,0 }), 1); EXPECT_EQ(y_v({ 1,2,0 }), 1); EXPECT_EQ(y_v({ 2,2,0 }), 1);
    }
}

TEST(Imgproc, HConcatenating)
{
    Image x({ 2, 2, 1 }, DataType::float32, "xyc", ColorType::GRAY);
    Image y;

    {
        View<DataType::float32> x_v(x);
        x_v({ 0,0,0 }) = 0; x_v({ 1,0,0 }) = 1;
        x_v({ 0,1,0 }) = 1; x_v({ 1,1,0 }) = 0;
        HConcat({ x, Neg(x) }, y);
        View<DataType::float32> y_v(y);

        // row 0
        EXPECT_FLOAT_EQ(y_v({ 0,0,0 }), 0); EXPECT_FLOAT_EQ(y_v({ 1,0,0 }), 1);
        EXPECT_FLOAT_EQ(y_v({ 2,0,0 }), 0); EXPECT_FLOAT_EQ(y_v({ 3,0,0 }), -1);
        // row 1
        EXPECT_FLOAT_EQ(y_v({ 0,1,0 }), 1); EXPECT_FLOAT_EQ(y_v({ 1,1,0 }), 0);
        EXPECT_FLOAT_EQ(y_v({ 2,1,0 }), -1); EXPECT_FLOAT_EQ(y_v({ 3,1,0 }), 0);
    }
    {
        // Test 1x1 Image
        x = Image({ 1, 1, 1 }, DataType::int32, "xyc", ColorType::GRAY);
        View<DataType::int32> x_v(x);

        x_v({ 0,0,0 }) = 1;
        HConcat({ x, Neg(x) }, y);
        View<DataType::int32> y_v(y);
        EXPECT_EQ(y_v({ 0,0,0 }), 1); EXPECT_EQ(y_v({ 1,0,0 }), -1);

        // Test 2x2 Image
        x = Image({ 2, 2, 1 }, DataType::int32, "xyc", ColorType::GRAY);
        x_v = x;
        x_v({ 0,0,0 }) = 1; x_v({ 1,0,0 }) = 1;
        x_v({ 0,1,0 }) = 1; x_v({ 1,1,0 }) = 1;
        HConcat({ x, Neg(x) }, y);
        y_v = y;
        EXPECT_EQ(y_v({ 0,0,0 }), 1); EXPECT_EQ(y_v({ 1,0,0 }), 1); EXPECT_EQ(y_v({ 2,0,0 }), -1); EXPECT_EQ(y_v({ 3,0,0 }), -1);
        EXPECT_EQ(y_v({ 0,1,0 }), 1); EXPECT_EQ(y_v({ 1,1,0 }), 1); EXPECT_EQ(y_v({ 2,1,0 }), -1); EXPECT_EQ(y_v({ 3,1,0 }), -1);
    }
}

TEST(Imgproc, VConcatenating)
{
    Image x({ 2, 2, 1 }, DataType::float32, "xyc", ColorType::GRAY);
    Image y;

    {
        View<DataType::float32> x_v(x);
        x_v({ 0,0,0 }) = 0; x_v({ 1,0,0 }) = 1;
        x_v({ 0,1,0 }) = 1; x_v({ 1,1,0 }) = 0;
        VConcat({ x, Neg(x) }, y);
        View<DataType::float32> y_v(y);

        EXPECT_FLOAT_EQ(y_v({ 0,0,0 }), 0); EXPECT_FLOAT_EQ(y_v({ 1,0,0 }), 1);
        EXPECT_FLOAT_EQ(y_v({ 0,1,0 }), 1); EXPECT_FLOAT_EQ(y_v({ 1,1,0 }), 0);

        EXPECT_FLOAT_EQ(y_v({ 0,2,0 }), 0); EXPECT_FLOAT_EQ(y_v({ 1,2,0 }), -1);
        EXPECT_FLOAT_EQ(y_v({ 0,3,0 }), -1); EXPECT_FLOAT_EQ(y_v({ 1,3,0 }), 0);
    }
    {
        // Test 1x1 Image
        x = Image({ 1, 1, 1 }, DataType::int32, "xyc", ColorType::GRAY);
        View<DataType::int32> x_v(x);

        x_v({ 0,0,0 }) = 1;
        VConcat({ x, Neg(x) }, y);
        View<DataType::int32> y_v(y);
        EXPECT_EQ(y_v({ 0,0,0 }), 1);
        EXPECT_EQ(y_v({ 0,1,0 }), -1);

        // Test 2x2 Image
        x = Image({ 2, 2, 1 }, DataType::int32, "xyc", ColorType::GRAY);
        x_v = x;
        x_v({ 0,0,0 }) = 1; x_v({ 1,0,0 }) = 1;
        x_v({ 0,1,0 }) = 1; x_v({ 1,1,0 }) = 1;
        VConcat({ x, Neg(x) }, y);
        y_v = y;
        EXPECT_EQ(y_v({ 0,0,0 }), 1); EXPECT_EQ(y_v({ 1,0,0 }), 1);
        EXPECT_EQ(y_v({ 0,1,0 }), 1); EXPECT_EQ(y_v({ 1,1,0 }), 1);
        EXPECT_EQ(y_v({ 0,2,0 }), -1); EXPECT_EQ(y_v({ 1,2,0 }), -1);
        EXPECT_EQ(y_v({ 0,3,0 }), -1); EXPECT_EQ(y_v({ 1,3,0 }), -1);
    }
}