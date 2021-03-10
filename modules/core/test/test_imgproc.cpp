/*
* ECVL - European Computer Vision Library
* Version: 0.3.1
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ecvl/core.h"

using namespace ecvl;

namespace
{
class TCore : public ::testing::Test
{
protected:
    Image out;

#define ECVL_TUPLE(type, ...) \
    Image g1_##type = Image({ 1, 1, 1 }, DataType::type, "xyc", ColorType::GRAY); \
    View<DataType::type> g1_##type##_v; \
    Image g2_##type = Image({ 2, 2, 1 }, DataType::type, "xyc", ColorType::GRAY); \
    View<DataType::type> g2_##type##_v; \
    Image rgb2_##type = Image({ 2, 2, 3 }, DataType::type, "xyc", ColorType::RGB); \
    View<DataType::type> rgb2_##type##_v; \

#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE

    void SetUp() override
    {
        out = Image();

    #define ECVL_TUPLE(type, ...) \
        g1_##type##_v = g1_##type; \
        g1_##type##_v({ 0,0,0 }) = 50; \
        \
        g2_##type##_v = g2_##type; \
        g2_##type##_v({ 0,0,0 }) = 50; g2_##type##_v({ 1,0,0 }) = 32; \
        g2_##type##_v({ 0,1,0 }) = 14; g2_##type##_v({ 1,1,0 }) = 60; \
        \
        rgb2_##type##_v = rgb2_##type; \
        rgb2_##type##_v({ 0,0,0 }) = 50; rgb2_##type##_v({ 1,0,0 }) = 32; \
        rgb2_##type##_v({ 0,1,0 }) = 14; rgb2_##type##_v({ 1,1,0 }) = 60; \
        rgb2_##type##_v({ 0,0,1 }) = 50; rgb2_##type##_v({ 1,0,1 }) = 32; \
        rgb2_##type##_v({ 0,1,1 }) = 14; rgb2_##type##_v({ 1,1,1 }) = 60; \
        rgb2_##type##_v({ 0,0,2 }) = 50; rgb2_##type##_v({ 1,0,2 }) = 32; \
        rgb2_##type##_v({ 0,1,2 }) = 14; rgb2_##type##_v({ 1,1,2 }) = 60; \

    #include "ecvl/core/datatype_existing_tuples.inc.h"
    #undef ECVL_TUPLE
    }
};

using Imgproc = TCore;

#define ECVL_TUPLE(type, ...) \
TEST_F(Imgproc, ResizeDim##type) \
{ \
    if (DataType::type != DataType::int64) { \
        ResizeDim(g1_##type, out, {1, 1}, InterpolationType::nearest); \
        View<DataType::type> out_v(out); \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
        EXPECT_THAT(out.dims_, testing::ElementsAre(1, 1, 1)); \
        \
        ResizeDim(g2_##type, out, {2, 1}, InterpolationType::nearest); \
        out_v = out; \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 32); \
        EXPECT_THAT(out.dims_, testing::ElementsAre(2, 1, 1)); \
        \
        ResizeDim(rgb2_##type, out, {2, 1}, InterpolationType::nearest); \
        out_v = out; \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 32); \
        EXPECT_TRUE(out_v({ 0,0,1 }) == 50); EXPECT_TRUE(out_v({ 1,0,1 }) == 32); \
        EXPECT_TRUE(out_v({ 0,0,2 }) == 50); EXPECT_TRUE(out_v({ 1,0,2 }) == 32); \
        EXPECT_THAT(out.dims_, testing::ElementsAre(2, 1, 3)); \
    } \
    else { \
        EXPECT_THROW(ResizeDim(g1_##type, out, {1, 1}, InterpolationType::nearest), std::runtime_error); \
    } \
} \
\
TEST_F(Imgproc, ResizeDimSameDst##type) \
{ \
    if (DataType::type != DataType::int64) { \
        ResizeDim(g1_##type, g1_##type, {1, 1}, InterpolationType::nearest); \
        View<DataType::type> out_v(g1_##type); \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
        EXPECT_THAT(out_v.dims_, testing::ElementsAre(1, 1, 1)); \
        \
        ResizeDim(g2_##type, g2_##type, {2, 1}, InterpolationType::nearest); \
        out_v = g2_##type; \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 32); \
        EXPECT_THAT(out_v.dims_, testing::ElementsAre(2, 1, 1)); \
        \
        ResizeDim(rgb2_##type, rgb2_##type, {2, 1}, InterpolationType::nearest); \
        out_v = rgb2_##type; \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 32); \
        EXPECT_TRUE(out_v({ 0,0,1 }) == 50); EXPECT_TRUE(out_v({ 1,0,1 }) == 32); \
        EXPECT_TRUE(out_v({ 0,0,2 }) == 50); EXPECT_TRUE(out_v({ 1,0,2 }) == 32); \
        EXPECT_THAT(out_v.dims_, testing::ElementsAre(2, 1, 3)); \
    } \
    else { \
        EXPECT_THROW(ResizeDim(g1_##type, g1_##type, {1, 1}, InterpolationType::nearest), std::runtime_error); \
    } \
} \
\
TEST_F(Imgproc, ResizeScale##type) \
{ \
    if (DataType::type != DataType::int64) { \
        ResizeScale(g1_##type, out, {1, 1}, InterpolationType::nearest); \
        View<DataType::type> out_v(out); \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
        EXPECT_THAT(out.dims_, testing::ElementsAre(1, 1, 1)); \
        \
        ResizeScale(g2_##type, out, {1, 0.5}, InterpolationType::nearest); \
        out_v = out; \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 32); \
        EXPECT_THAT(out.dims_, testing::ElementsAre(2, 1, 1)); \
        \
        ResizeScale(rgb2_##type, out, {1, 0.5}, InterpolationType::nearest); \
        out_v = out; \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 32); \
        EXPECT_TRUE(out_v({ 0,0,1 }) == 50); EXPECT_TRUE(out_v({ 1,0,1 }) == 32); \
        EXPECT_TRUE(out_v({ 0,0,2 }) == 50); EXPECT_TRUE(out_v({ 1,0,2 }) == 32); \
        EXPECT_THAT(out.dims_, testing::ElementsAre(2, 1, 3)); \
    } \
    else { \
        EXPECT_THROW(ResizeScale(g1_##type, out, {1, 1}, InterpolationType::nearest), std::runtime_error); \
    } \
} \
\
TEST_F(Imgproc, ResizeScaleSameDst##type) \
{ \
    if (DataType::type != DataType::int64) { \
        ResizeScale(g1_##type, g1_##type, {1, 1}, InterpolationType::nearest); \
        View<DataType::type> out_v(g1_##type); \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
        EXPECT_THAT(out_v.dims_, testing::ElementsAre(1, 1, 1)); \
        \
        ResizeScale(g2_##type, g2_##type, {1, 0.5}, InterpolationType::nearest); \
        out_v = g2_##type; \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 32); \
        EXPECT_THAT(out_v.dims_, testing::ElementsAre(2, 1, 1)); \
        \
        ResizeScale(rgb2_##type, rgb2_##type, {1, 0.5}, InterpolationType::nearest); \
        out_v = rgb2_##type; \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 32); \
        EXPECT_TRUE(out_v({ 0,0,1 }) == 50); EXPECT_TRUE(out_v({ 1,0,1 }) == 32); \
        EXPECT_TRUE(out_v({ 0,0,2 }) == 50); EXPECT_TRUE(out_v({ 1,0,2 }) == 32); \
        EXPECT_THAT(out_v.dims_, testing::ElementsAre(2, 1, 3)); \
    } \
    else { \
        EXPECT_THROW(ResizeScale(g1_##type, g1_##type, {1, 1}, InterpolationType::nearest), std::runtime_error); \
    } \
} \
\
TEST_F(Imgproc, Flip2D##type) \
{ \
    Flip2D(g1_##type, out); \
    View<DataType::type> out_v(out); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
    \
    Flip2D(g2_##type, out); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 14); EXPECT_TRUE(out_v({ 1,0,0 }) == 60); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 50); EXPECT_TRUE(out_v({ 1,1,0 }) == 32); \
    \
    Flip2D(rgb2_##type, out); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 14); EXPECT_TRUE(out_v({ 1,0,0 }) == 60); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 50); EXPECT_TRUE(out_v({ 1,1,0 }) == 32); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == 14); EXPECT_TRUE(out_v({ 1,0,1 }) == 60); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == 50); EXPECT_TRUE(out_v({ 1,1,1 }) == 32); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == 14); EXPECT_TRUE(out_v({ 1,0,2 }) == 60); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == 50); EXPECT_TRUE(out_v({ 1,1,2 }) == 32); \
} \
\
TEST_F(Imgproc, Flip2DSameDst##type) \
{ \
    Flip2D(g1_##type, g1_##type); \
    View<DataType::type> out_v(g1_##type); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
    \
    Flip2D(g2_##type, g2_##type); \
    out_v = g2_##type; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 14); EXPECT_TRUE(out_v({ 1,0,0 }) == 60); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 50); EXPECT_TRUE(out_v({ 1,1,0 }) == 32); \
    \
    Flip2D(rgb2_##type, rgb2_##type); \
    out_v = rgb2_##type; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 14); EXPECT_TRUE(out_v({ 1,0,0 }) == 60); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 50); EXPECT_TRUE(out_v({ 1,1,0 }) == 32); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == 14); EXPECT_TRUE(out_v({ 1,0,1 }) == 60); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == 50); EXPECT_TRUE(out_v({ 1,1,1 }) == 32); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == 14); EXPECT_TRUE(out_v({ 1,0,2 }) == 60); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == 50); EXPECT_TRUE(out_v({ 1,1,2 }) == 32); \
} \
\
TEST_F(Imgproc, Mirror2D##type) \
{ \
    Mirror2D(g1_##type, out); \
    View<DataType::type> out_v(out); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
    \
    Mirror2D(g2_##type, out); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 32); EXPECT_TRUE(out_v({ 1,0,0 }) == 50); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 60); EXPECT_TRUE(out_v({ 1,1,0 }) == 14); \
    \
    Mirror2D(rgb2_##type, out); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 32); EXPECT_TRUE(out_v({ 1,0,0 }) == 50); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 60); EXPECT_TRUE(out_v({ 1,1,0 }) == 14); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == 32); EXPECT_TRUE(out_v({ 1,0,1 }) == 50); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == 60); EXPECT_TRUE(out_v({ 1,1,1 }) == 14); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == 32); EXPECT_TRUE(out_v({ 1,0,2 }) == 50); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == 60); EXPECT_TRUE(out_v({ 1,1,2 }) == 14); \
} \
\
TEST_F(Imgproc, Mirror2DSameDst##type) \
{ \
    Mirror2D(g1_##type, g1_##type); \
    View<DataType::type> out_v(g1_##type); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
    \
    Mirror2D(g2_##type, g2_##type); \
    out_v = g2_##type; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 32); EXPECT_TRUE(out_v({ 1,0,0 }) == 50); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 60); EXPECT_TRUE(out_v({ 1,1,0 }) == 14); \
    \
    Mirror2D(rgb2_##type, rgb2_##type); \
    out_v = rgb2_##type; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 32); EXPECT_TRUE(out_v({ 1,0,0 }) == 50); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 60); EXPECT_TRUE(out_v({ 1,1,0 }) == 14); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == 32); EXPECT_TRUE(out_v({ 1,0,1 }) == 50); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == 60); EXPECT_TRUE(out_v({ 1,1,1 }) == 14); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == 32); EXPECT_TRUE(out_v({ 1,0,2 }) == 50); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == 60); EXPECT_TRUE(out_v({ 1,1,2 }) == 14); \
} \
\
TEST_F(Imgproc, Threshold##type) \
{ \
    Threshold(g1_##type, out, 35, 127, ThresholdingType::BINARY); \
    View<DataType::type> out_v(out); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 127); \
    \
    Threshold(g1_##type, out, 35, 127, ThresholdingType::BINARY_INV); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 0); \
    \
    Threshold(g2_##type, out, 35, 127, ThresholdingType::BINARY); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 127); EXPECT_TRUE(out_v({ 1,0,0 }) == 0); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 0); EXPECT_TRUE(out_v({ 1,1,0 }) == 127); \
    \
    Threshold(g2_##type, out, 35, 127, ThresholdingType::BINARY_INV); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 0); EXPECT_TRUE(out_v({ 1,0,0 }) == 127); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 127); EXPECT_TRUE(out_v({ 1,1,0 }) == 0); \
    \
    Threshold(rgb2_##type, out, 35, 127, ThresholdingType::BINARY); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 127); EXPECT_TRUE(out_v({ 1,0,0 }) == 0); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 0); EXPECT_TRUE(out_v({ 1,1,0 }) == 127); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == 127); EXPECT_TRUE(out_v({ 1,0,1 }) == 0); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == 0); EXPECT_TRUE(out_v({ 1,1,1 }) == 127); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == 127); EXPECT_TRUE(out_v({ 1,0,2 }) == 0); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == 0); EXPECT_TRUE(out_v({ 1,1,2 }) == 127); \
    \
    Threshold(rgb2_##type, out, 35, 127, ThresholdingType::BINARY_INV); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 0); EXPECT_TRUE(out_v({ 1,0,0 }) == 127); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 127); EXPECT_TRUE(out_v({ 1,1,0 }) == 0); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == 0); EXPECT_TRUE(out_v({ 1,0,1 }) == 127); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == 127); EXPECT_TRUE(out_v({ 1,1,1 }) == 0); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == 0); EXPECT_TRUE(out_v({ 1,0,2 }) == 127); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == 127); EXPECT_TRUE(out_v({ 1,1,2 }) == 0); \
} \
\
TEST_F(Imgproc, ThresholdSameDst##type) \
{ \
    Threshold(g1_##type, g1_##type, 35, 127, ThresholdingType::BINARY); \
    View<DataType::type> out_v(g1_##type); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 127); \
    \
    Threshold(g1_##type, g1_##type, 35, 127, ThresholdingType::BINARY_INV); \
    out_v = g1_##type; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 0); \
    \
    Threshold(g2_##type, g2_##type, 35, 127, ThresholdingType::BINARY); \
    out_v = g2_##type; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 127); EXPECT_TRUE(out_v({ 1,0,0 }) == 0); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 0); EXPECT_TRUE(out_v({ 1,1,0 }) == 127); \
    \
    Threshold(g2_##type, g2_##type, 35, 127, ThresholdingType::BINARY_INV); \
    out_v = g2_##type; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 0); EXPECT_TRUE(out_v({ 1,0,0 }) == 127); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 127); EXPECT_TRUE(out_v({ 1,1,0 }) == 0); \
    \
    Threshold(rgb2_##type, rgb2_##type, 35, 127, ThresholdingType::BINARY); \
    out_v = rgb2_##type; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 127); EXPECT_TRUE(out_v({ 1,0,0 }) == 0); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 0); EXPECT_TRUE(out_v({ 1,1,0 }) == 127); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == 127); EXPECT_TRUE(out_v({ 1,0,1 }) == 0); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == 0); EXPECT_TRUE(out_v({ 1,1,1 }) == 127); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == 127); EXPECT_TRUE(out_v({ 1,0,2 }) == 0); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == 0); EXPECT_TRUE(out_v({ 1,1,2 }) == 127); \
    \
    Threshold(rgb2_##type, rgb2_##type, 35, 127, ThresholdingType::BINARY_INV); \
    out_v = rgb2_##type; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 0); EXPECT_TRUE(out_v({ 1,0,0 }) == 127); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 127); EXPECT_TRUE(out_v({ 1,1,0 }) == 0); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == 0); EXPECT_TRUE(out_v({ 1,0,1 }) == 127); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == 127); EXPECT_TRUE(out_v({ 1,1,1 }) == 0); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == 0); EXPECT_TRUE(out_v({ 1,0,2 }) == 127); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == 127); EXPECT_TRUE(out_v({ 1,1,2 }) == 0); \
} \
\
TEST_F(Imgproc, HConcat##type) \
{ \
    HConcat({ g1_##type, g1_##type }, out); \
    View<DataType::type> out_v(out); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 50); \
    \
    HConcat({ g2_##type, g2_##type }, out); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) = 32); EXPECT_TRUE(out_v({ 2,0,0 }) == 50); EXPECT_TRUE(out_v({ 3,0,0 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 14); EXPECT_TRUE(out_v({ 1,1,0 }) = 60); EXPECT_TRUE(out_v({ 2,1,0 }) == 14); EXPECT_TRUE(out_v({ 3,1,0 }) = 60); \
    \
    HConcat({ rgb2_##type, rgb2_##type }, out); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) = 32); EXPECT_TRUE(out_v({ 2,0,0 }) == 50); EXPECT_TRUE(out_v({ 3,0,0 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 14); EXPECT_TRUE(out_v({ 1,1,0 }) = 60); EXPECT_TRUE(out_v({ 2,1,0 }) == 14); EXPECT_TRUE(out_v({ 3,1,0 }) = 60); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == 50); EXPECT_TRUE(out_v({ 1,0,1 }) = 32); EXPECT_TRUE(out_v({ 2,0,1 }) == 50); EXPECT_TRUE(out_v({ 3,0,1 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == 14); EXPECT_TRUE(out_v({ 1,1,1 }) = 60); EXPECT_TRUE(out_v({ 2,1,1 }) == 14); EXPECT_TRUE(out_v({ 3,1,1 }) = 60); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == 50); EXPECT_TRUE(out_v({ 1,0,2 }) = 32); EXPECT_TRUE(out_v({ 2,0,2 }) == 50); EXPECT_TRUE(out_v({ 3,0,2 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == 14); EXPECT_TRUE(out_v({ 1,1,2 }) = 60); EXPECT_TRUE(out_v({ 2,1,2 }) == 14); EXPECT_TRUE(out_v({ 3,1,2 }) = 60); \
} \
\
TEST_F(Imgproc, HConcatSameDst##type) \
{ \
    HConcat({ g1_##type, g1_##type }, g1_##type); \
    View<DataType::type> out_v(g1_##type); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 50); \
    \
    HConcat({ g2_##type, g2_##type }, g2_##type); \
    out_v = g2_##type; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) = 32); EXPECT_TRUE(out_v({ 2,0,0 }) == 50); EXPECT_TRUE(out_v({ 3,0,0 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 14); EXPECT_TRUE(out_v({ 1,1,0 }) = 60); EXPECT_TRUE(out_v({ 2,1,0 }) == 14); EXPECT_TRUE(out_v({ 3,1,0 }) = 60); \
    \
    HConcat({ rgb2_##type, rgb2_##type }, rgb2_##type); \
    out_v = rgb2_##type; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) = 32); EXPECT_TRUE(out_v({ 2,0,0 }) == 50); EXPECT_TRUE(out_v({ 3,0,0 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 14); EXPECT_TRUE(out_v({ 1,1,0 }) = 60); EXPECT_TRUE(out_v({ 2,1,0 }) == 14); EXPECT_TRUE(out_v({ 3,1,0 }) = 60); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == 50); EXPECT_TRUE(out_v({ 1,0,1 }) = 32); EXPECT_TRUE(out_v({ 2,0,1 }) == 50); EXPECT_TRUE(out_v({ 3,0,1 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == 14); EXPECT_TRUE(out_v({ 1,1,1 }) = 60); EXPECT_TRUE(out_v({ 2,1,1 }) == 14); EXPECT_TRUE(out_v({ 3,1,1 }) = 60); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == 50); EXPECT_TRUE(out_v({ 1,0,2 }) = 32); EXPECT_TRUE(out_v({ 2,0,2 }) == 50); EXPECT_TRUE(out_v({ 3,0,2 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == 14); EXPECT_TRUE(out_v({ 1,1,2 }) = 60); EXPECT_TRUE(out_v({ 2,1,2 }) == 14); EXPECT_TRUE(out_v({ 3,1,2 }) = 60); \
} \
\
TEST_F(Imgproc, VConcat##type) \
{ \
    VConcat({ g1_##type, g1_##type }, out); \
    View<DataType::type> out_v(out); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 50); \
    \
    VConcat({ g2_##type, g2_##type }, out); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 14); EXPECT_TRUE(out_v({ 1,1,0 }) = 60); \
    EXPECT_TRUE(out_v({ 0,2,0 }) == 50); EXPECT_TRUE(out_v({ 1,2,0 }) = 32); \
    EXPECT_TRUE(out_v({ 0,3,0 }) == 14); EXPECT_TRUE(out_v({ 1,3,0 }) = 60); \
    \
    VConcat({ rgb2_##type, rgb2_##type }, out); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 14); EXPECT_TRUE(out_v({ 1,1,0 }) = 60); \
    EXPECT_TRUE(out_v({ 0,2,0 }) == 50); EXPECT_TRUE(out_v({ 1,2,0 }) = 32); \
    EXPECT_TRUE(out_v({ 0,3,0 }) == 14); EXPECT_TRUE(out_v({ 1,3,0 }) = 60); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == 50); EXPECT_TRUE(out_v({ 1,0,1 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == 14); EXPECT_TRUE(out_v({ 1,1,1 }) = 60); \
    EXPECT_TRUE(out_v({ 0,2,1 }) == 50); EXPECT_TRUE(out_v({ 1,2,1 }) = 32); \
    EXPECT_TRUE(out_v({ 0,3,1 }) == 14); EXPECT_TRUE(out_v({ 1,3,1 }) = 60); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == 50); EXPECT_TRUE(out_v({ 1,0,2 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == 14); EXPECT_TRUE(out_v({ 1,1,2 }) = 60); \
    EXPECT_TRUE(out_v({ 0,2,2 }) == 50); EXPECT_TRUE(out_v({ 1,2,2 }) = 32); \
    EXPECT_TRUE(out_v({ 0,3,2 }) == 14); EXPECT_TRUE(out_v({ 1,3,2 }) = 60); \
} \
\
TEST_F(Imgproc, VConcatSameDst##type) \
{ \
    VConcat({ g1_##type, g1_##type }, g1_##type); \
    View<DataType::type> out_v(g1_##type); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 50); \
    \
    VConcat({ g2_##type, g2_##type },  g2_##type); \
    out_v =  g2_##type; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 14); EXPECT_TRUE(out_v({ 1,1,0 }) = 60); \
    EXPECT_TRUE(out_v({ 0,2,0 }) == 50); EXPECT_TRUE(out_v({ 1,2,0 }) = 32); \
    EXPECT_TRUE(out_v({ 0,3,0 }) == 14); EXPECT_TRUE(out_v({ 1,3,0 }) = 60); \
    \
    VConcat({ rgb2_##type, rgb2_##type }, rgb2_##type); \
    out_v = rgb2_##type; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 14); EXPECT_TRUE(out_v({ 1,1,0 }) = 60); \
    EXPECT_TRUE(out_v({ 0,2,0 }) == 50); EXPECT_TRUE(out_v({ 1,2,0 }) = 32); \
    EXPECT_TRUE(out_v({ 0,3,0 }) == 14); EXPECT_TRUE(out_v({ 1,3,0 }) = 60); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == 50); EXPECT_TRUE(out_v({ 1,0,1 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == 14); EXPECT_TRUE(out_v({ 1,1,1 }) = 60); \
    EXPECT_TRUE(out_v({ 0,2,1 }) == 50); EXPECT_TRUE(out_v({ 1,2,1 }) = 32); \
    EXPECT_TRUE(out_v({ 0,3,1 }) == 14); EXPECT_TRUE(out_v({ 1,3,1 }) = 60); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == 50); EXPECT_TRUE(out_v({ 1,0,2 }) = 32); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == 14); EXPECT_TRUE(out_v({ 1,1,2 }) = 60); \
    EXPECT_TRUE(out_v({ 0,2,2 }) == 50); EXPECT_TRUE(out_v({ 1,2,2 }) = 32); \
    EXPECT_TRUE(out_v({ 0,3,2 }) == 14); EXPECT_TRUE(out_v({ 1,3,2 }) = 60); \
} \
\
TEST_F(Imgproc, Transpose##type) \
{ \
    Transpose(g1_##type, out); \
    View<DataType::type> out_v(out); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
    \
    Transpose(g2_##type, out); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 14); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 32); EXPECT_TRUE(out_v({ 1,1,0 }) == 60); \
    \
    Transpose(rgb2_##type, out); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 14); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 32); EXPECT_TRUE(out_v({ 1,1,0 }) == 60); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == 50); EXPECT_TRUE(out_v({ 1,0,1 }) == 14); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == 32); EXPECT_TRUE(out_v({ 1,1,1 }) == 60); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == 50); EXPECT_TRUE(out_v({ 1,0,2 }) == 14); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == 32); EXPECT_TRUE(out_v({ 1,1,2 }) == 60); \
} \
\
TEST_F(Imgproc, TransposeSameDst##type) \
{ \
    Transpose(g1_##type, g1_##type); \
    View<DataType::type> out_v(g1_##type); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
    \
    Transpose(g2_##type, g2_##type); \
    out_v = g2_##type; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 14); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 32); EXPECT_TRUE(out_v({ 1,1,0 }) == 60); \
    \
    Transpose(rgb2_##type, rgb2_##type); \
    out_v = rgb2_##type; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 14); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 32); EXPECT_TRUE(out_v({ 1,1,0 }) == 60); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == 50); EXPECT_TRUE(out_v({ 1,0,1 }) == 14); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == 32); EXPECT_TRUE(out_v({ 1,1,1 }) == 60); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == 50); EXPECT_TRUE(out_v({ 1,0,2 }) == 14); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == 32); EXPECT_TRUE(out_v({ 1,1,2 }) == 60); \
} \
\
TEST_F(Imgproc, GridDistortion##type) \
{ \
    if (DataType::type != DataType::int64) { \
        GridDistortion(g1_##type, out); \
        View<DataType::type> out_v(out); \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
        \
        GridDistortion(g2_##type, out); \
        out_v = out; \
        EXPECT_EQ(g2_##type.dims_, out_v.dims_); \
        EXPECT_EQ(g2_##type.colortype_, out_v.colortype_); \
        EXPECT_EQ(g2_##type.elemtype_, out_v.elemtype_); \
        EXPECT_EQ(g2_##type.channels_, out_v.channels_); \
        \
        GridDistortion(rgb2_##type, out); \
        out_v = out; \
        EXPECT_EQ(rgb2_##type.dims_, out_v.dims_); \
        EXPECT_EQ(rgb2_##type.colortype_, out_v.colortype_); \
        EXPECT_EQ(rgb2_##type.elemtype_, out_v.elemtype_); \
        EXPECT_EQ(rgb2_##type.channels_, out_v.channels_); \
    } \
    else { \
        EXPECT_THROW(GridDistortion(g1_##type, out), std::runtime_error); \
    } \
} \
\
TEST_F(Imgproc, GridDistortionSameDst##type) \
{ \
    if (DataType::type != DataType::int64) { \
        GridDistortion(g1_##type, g1_##type); \
        View<DataType::type> out_v(g1_##type); \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
        \
        GridDistortion(g2_##type, g2_##type); \
        out_v = g2_##type; \
        EXPECT_EQ(g2_##type.dims_, out_v.dims_); \
        EXPECT_EQ(g2_##type.colortype_, out_v.colortype_); \
        EXPECT_EQ(g2_##type.elemtype_, out_v.elemtype_); \
        EXPECT_EQ(g2_##type.channels_, out_v.channels_); \
        \
        GridDistortion(rgb2_##type, rgb2_##type); \
        out_v = rgb2_##type; \
        EXPECT_EQ(rgb2_##type.dims_, out_v.dims_); \
        EXPECT_EQ(rgb2_##type.colortype_, out_v.colortype_); \
        EXPECT_EQ(rgb2_##type.elemtype_, out_v.elemtype_); \
        EXPECT_EQ(rgb2_##type.channels_, out_v.channels_); \
    } \
    else { \
        EXPECT_THROW(GridDistortion(g1_##type, g1_##type), std::runtime_error); \
    } \
} \
\
TEST_F(Imgproc, ElasticTransform##type) \
{ \
    if (DataType::type != DataType::int64) { \
        ElasticTransform(g1_##type, out); \
        View<DataType::type> out_v(out); \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
        \
        ElasticTransform(g2_##type, out); \
        out_v = out; \
        EXPECT_EQ(g2_##type.dims_, out_v.dims_); \
        EXPECT_EQ(g2_##type.colortype_, out_v.colortype_); \
        EXPECT_EQ(g2_##type.elemtype_, out_v.elemtype_); \
        EXPECT_EQ(g2_##type.channels_, out_v.channels_); \
        \
        ElasticTransform(rgb2_##type, out); \
        out_v = out; \
        EXPECT_EQ(rgb2_##type.dims_, out_v.dims_); \
        EXPECT_EQ(rgb2_##type.colortype_, out_v.colortype_); \
        EXPECT_EQ(rgb2_##type.elemtype_, out_v.elemtype_); \
        EXPECT_EQ(rgb2_##type.channels_, out_v.channels_); \
    } \
    else { \
        EXPECT_THROW(ElasticTransform(g1_##type, out), std::runtime_error); \
    } \
} \
\
TEST_F(Imgproc, ElasticTransformSameDst##type) \
{ \
    if (DataType::type != DataType::int64) { \
        ElasticTransform(g1_##type, g1_##type); \
        View<DataType::type> out_v(g1_##type); \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
        \
        ElasticTransform(g2_##type, g2_##type); \
        out_v = g2_##type; \
        EXPECT_EQ(g2_##type.dims_, out_v.dims_); \
        EXPECT_EQ(g2_##type.colortype_, out_v.colortype_); \
        EXPECT_EQ(g2_##type.elemtype_, out_v.elemtype_); \
        EXPECT_EQ(g2_##type.channels_, out_v.channels_); \
        \
        ElasticTransform(rgb2_##type, rgb2_##type); \
        out_v = rgb2_##type; \
        EXPECT_EQ(rgb2_##type.dims_, out_v.dims_); \
        EXPECT_EQ(rgb2_##type.colortype_, out_v.colortype_); \
        EXPECT_EQ(rgb2_##type.elemtype_, out_v.elemtype_); \
        EXPECT_EQ(rgb2_##type.channels_, out_v.channels_); \
    } \
    else { \
        EXPECT_THROW(ElasticTransform(g1_##type, g1_##type), std::runtime_error); \
    } \
} \
\
TEST_F(Imgproc, OpticalDistortion##type) \
{ \
    if (DataType::type != DataType::int64) { \
        OpticalDistortion(g1_##type, out); \
        View<DataType::type> out_v(out); \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
        \
        OpticalDistortion(g2_##type, out); \
        out_v = out; \
        EXPECT_EQ(g2_##type.dims_, out_v.dims_); \
        EXPECT_EQ(g2_##type.colortype_, out_v.colortype_); \
        EXPECT_EQ(g2_##type.elemtype_, out_v.elemtype_); \
        EXPECT_EQ(g2_##type.channels_, out_v.channels_); \
        \
        OpticalDistortion(rgb2_##type, out); \
        out_v = out; \
        EXPECT_EQ(rgb2_##type.dims_, out_v.dims_); \
        EXPECT_EQ(rgb2_##type.colortype_, out_v.colortype_); \
        EXPECT_EQ(rgb2_##type.elemtype_, out_v.elemtype_); \
        EXPECT_EQ(rgb2_##type.channels_, out_v.channels_); \
    } \
    else { \
        EXPECT_THROW(OpticalDistortion(g1_##type, out), std::runtime_error); \
    } \
} \
\
TEST_F(Imgproc, OpticalDistortionSameDst##type) \
{ \
    if (DataType::type != DataType::int64) { \
        OpticalDistortion(g1_##type, g1_##type); \
        View<DataType::type> out_v(g1_##type); \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
        \
        OpticalDistortion(g2_##type, g2_##type); \
        out_v = g2_##type; \
        EXPECT_EQ(g2_##type.dims_, out_v.dims_); \
        EXPECT_EQ(g2_##type.colortype_, out_v.colortype_); \
        EXPECT_EQ(g2_##type.elemtype_, out_v.elemtype_); \
        EXPECT_EQ(g2_##type.channels_, out_v.channels_); \
        \
        OpticalDistortion(rgb2_##type, rgb2_##type); \
        out_v = rgb2_##type; \
        EXPECT_EQ(rgb2_##type.dims_, out_v.dims_); \
        EXPECT_EQ(rgb2_##type.colortype_, out_v.colortype_); \
        EXPECT_EQ(rgb2_##type.elemtype_, out_v.elemtype_); \
        EXPECT_EQ(rgb2_##type.channels_, out_v.channels_); \
    } \
    else { \
        EXPECT_THROW(OpticalDistortion(g1_##type, g1_##type), std::runtime_error); \
    } \
}\
TEST_F(Imgproc, Salt##type) \
{ \
    Salt(g1_##type, out, 0.5, false, 0); \
    View<DataType::type> out_v(out); \
    EXPECT_EQ(g1_##type.dims_, out_v.dims_); \
    EXPECT_EQ(g1_##type.colortype_, out_v.colortype_); \
    EXPECT_EQ(g1_##type.elemtype_, out_v.elemtype_); \
    EXPECT_EQ(g1_##type.channels_, out_v.channels_); \
    \
    Salt(g2_##type, out, 0.5, false, 0); \
    out_v = out; \
    EXPECT_EQ(g2_##type.dims_, out_v.dims_); \
    EXPECT_EQ(g2_##type.colortype_, out_v.colortype_); \
    EXPECT_EQ(g2_##type.elemtype_, out_v.elemtype_); \
    EXPECT_EQ(g2_##type.channels_, out_v.channels_); \
    \
    Salt(rgb2_##type, out, 0.5, false, 0); \
    out_v = out; \
    EXPECT_EQ(rgb2_##type.dims_, out_v.dims_); \
    EXPECT_EQ(rgb2_##type.colortype_, out_v.colortype_); \
    EXPECT_EQ(rgb2_##type.elemtype_, out_v.elemtype_); \
    EXPECT_EQ(rgb2_##type.channels_, out_v.channels_); \
} \
\
TEST_F(Imgproc, SaltSameDst##type) \
{ \
    Salt(g1_##type, g1_##type, 0.5, false, 0); \
    View<DataType::type> out_v(g1_##type); \
    EXPECT_EQ(g1_##type.dims_, out_v.dims_); \
    EXPECT_EQ(g1_##type.colortype_, out_v.colortype_); \
    EXPECT_EQ(g1_##type.elemtype_, out_v.elemtype_); \
    EXPECT_EQ(g1_##type.channels_, out_v.channels_); \
    \
    Salt(g2_##type, g2_##type, 0.5, false, 0); \
    out_v = g2_##type; \
    EXPECT_EQ(g2_##type.dims_, out_v.dims_); \
    EXPECT_EQ(g2_##type.colortype_, out_v.colortype_); \
    EXPECT_EQ(g2_##type.elemtype_, out_v.elemtype_); \
    EXPECT_EQ(g2_##type.channels_, out_v.channels_); \
    \
    Salt(rgb2_##type, rgb2_##type, 0.5, false, 0); \
    out_v = rgb2_##type; \
    EXPECT_EQ(rgb2_##type.dims_, out_v.dims_); \
    EXPECT_EQ(rgb2_##type.colortype_, out_v.colortype_); \
    EXPECT_EQ(rgb2_##type.elemtype_, out_v.elemtype_); \
    EXPECT_EQ(rgb2_##type.channels_, out_v.channels_); \
} \
\
TEST_F(Imgproc, CCLSameDst##type) \
{ \
    if (DataType::type == DataType::uint8) { \
        Threshold(g1_##type, g1_##type, 35, 255, ThresholdingType::BINARY); \
        ConnectedComponentsLabeling(g1_##type, g1_##type); \
        View<DataType::int32> out_v(g1_##type); /* CCL output image is fixed to int32 */ \
        EXPECT_EQ(g1_##type.dims_, out_v.dims_); \
        EXPECT_EQ(g1_##type.colortype_, out_v.colortype_); \
        /*EXPECT_EQ(g1_##type.elemtype_, out_v.elemtype_);*/ \
        EXPECT_EQ(g1_##type.channels_, out_v.channels_); \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 1); \
        \
        Threshold(g2_##type, g2_##type, 35, 255, ThresholdingType::BINARY); \
        ConnectedComponentsLabeling(g2_##type, g2_##type); \
        out_v = g2_##type; \
        EXPECT_EQ(g2_##type.dims_, out_v.dims_); \
        EXPECT_EQ(g2_##type.colortype_, out_v.colortype_); \
        /*EXPECT_EQ(g2_##type.elemtype_, out_v.elemtype_);*/ \
        EXPECT_EQ(g2_##type.channels_, out_v.channels_); \
        EXPECT_TRUE(out_v({ 0,0,0 }) == 1); EXPECT_TRUE(out_v({ 1,0,0 }) == 0); \
        EXPECT_TRUE(out_v({ 0,1,0 }) == 0); EXPECT_TRUE(out_v({ 1,1,0 }) == 1); \
    } \
    else { \
        EXPECT_THROW(ConnectedComponentsLabeling(g1_##type, g1_##type), std::runtime_error); \
    } \
} \
\
TEST_F(Imgproc, Normalize##type) \
{ \
    constexpr double mean = 39.0; \
    constexpr double std = 17.5783958312469; \
    Normalize(g1_##type, out, mean, std); \
    View<DataType::type> out_v(out); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == saturate_cast<TypeInfo_t<DataType::type>>((g1_##type##_v({ 0,0,0 }) - mean) / std)); \
    \
    Normalize(g2_##type, out, mean, std); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == saturate_cast<TypeInfo_t<DataType::type>>((g2_##type##_v({ 0,0,0 }) - mean) / std)); \
    EXPECT_TRUE(out_v({ 1,0,0 }) == saturate_cast<TypeInfo_t<DataType::type>>((g2_##type##_v({ 1,0,0 }) - mean) / std)); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == saturate_cast<TypeInfo_t<DataType::type>>((g2_##type##_v({ 0,1,0 }) - mean) / std)); \
    EXPECT_TRUE(out_v({ 1,1,0 }) == saturate_cast<TypeInfo_t<DataType::type>>((g2_##type##_v({ 1,1,0 }) - mean) / std)); \
    \
    Normalize(rgb2_##type, out, mean, std); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 0,0,0 }) - mean) / std)); \
    EXPECT_TRUE(out_v({ 1,0,0 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 1,0,0 }) - mean) / std)); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 0,1,0 }) - mean) / std)); \
    EXPECT_TRUE(out_v({ 1,1,0 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 1,1,0 }) - mean) / std)); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 0,0,1 }) - mean) / std)); \
    EXPECT_TRUE(out_v({ 1,0,1 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 1,0,1 }) - mean) / std)); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 0,1,1 }) - mean) / std)); \
    EXPECT_TRUE(out_v({ 1,1,1 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 1,1,1 }) - mean) / std)); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 0,0,2 }) - mean) / std)); \
    EXPECT_TRUE(out_v({ 1,0,2 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 1,0,2 }) - mean) / std)); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 0,1,2 }) - mean) / std)); \
    EXPECT_TRUE(out_v({ 1,1,2 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 1,1,2 }) - mean) / std)); \
} \
TEST_F(Imgproc, NormalizeChannels##type) \
{ \
    const std::vector<double> mean = { 39.0, 38.0, 33.0 }; \
    const std::vector<double> std = { 17.5783958312469, 14.5783958312469, 23.5783958312469 }; \
    Normalize(rgb2_##type, out, mean, std); \
    View<DataType::type> out_v(out); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 0,0,0 }) - mean[0]) / std[0])); \
    EXPECT_TRUE(out_v({ 1,0,0 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 1,0,0 }) - mean[0]) / std[0])); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 0,1,0 }) - mean[0]) / std[0])); \
    EXPECT_TRUE(out_v({ 1,1,0 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 1,1,0 }) - mean[0]) / std[0])); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 0,0,1 }) - mean[1]) / std[1])); \
    EXPECT_TRUE(out_v({ 1,0,1 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 1,0,1 }) - mean[1]) / std[1])); \
    EXPECT_TRUE(out_v({ 0,1,1 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 0,1,1 }) - mean[1]) / std[1])); \
    EXPECT_TRUE(out_v({ 1,1,1 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 1,1,1 }) - mean[1]) / std[1])); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 0,0,2 }) - mean[2]) / std[2])); \
    EXPECT_TRUE(out_v({ 1,0,2 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 1,0,2 }) - mean[2]) / std[2])); \
    EXPECT_TRUE(out_v({ 0,1,2 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 0,1,2 }) - mean[2]) / std[2])); \
    EXPECT_TRUE(out_v({ 1,1,2 }) == saturate_cast<TypeInfo_t<DataType::type>>((rgb2_##type##_v({ 1,1,2 }) - mean[2]) / std[2])); \
} \
\
TEST_F(Imgproc, CenterCrop##type) \
{ \
    std::vector<int> size { 1,1 }; \
    CenterCrop(g1_##type, out, size); \
    View<DataType::type> out_v(out); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
    EXPECT_THAT(out.dims_, testing::ElementsAre(1, 1, 1)); \
    \
    CenterCrop(g2_##type, out, size); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
    EXPECT_THAT(out.dims_, testing::ElementsAre(1, 1, 1)); \
    \
    CenterCrop(rgb2_##type, out, size); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
    EXPECT_TRUE(out_v({ 0,0,1 }) == 50); \
    EXPECT_TRUE(out_v({ 0,0,2 }) == 50); \
    EXPECT_THAT(out.dims_, testing::ElementsAre(1, 1, 3)); \
} \
\
TEST_F(Imgproc, ScaleTo_##type) \
{ \
    ScaleTo(g2_##type, out, 14, 60); \
    View<DataType::type> out_v(out); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 32); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 14); EXPECT_TRUE(out_v({ 1,1,0 }) == 60); \
    ScaleTo(g2_##type, out, 0, 1); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == static_cast<TypeInfo_t<DataType::type>>((g2_##type##_v({ 0,0,0 }) * (1 / 46.) + (1 - ((1 / 46.) * 60))))); \
    EXPECT_TRUE(out_v({ 1,0,0 }) == static_cast<TypeInfo_t<DataType::type>>((g2_##type##_v({ 1,0,0 }) * (1 / 46.) + (1 - ((1 / 46.) * 60))))); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == static_cast<TypeInfo_t<DataType::type>>((g2_##type##_v({ 0,1,0 }) * (1 / 46.) + (1 - ((1 / 46.) * 60))))); \
    EXPECT_TRUE(out_v({ 1,1,0 }) == static_cast<TypeInfo_t<DataType::type>>((g2_##type##_v({ 1,1,0 }) * (1 / 46.) + (1 - ((1 / 46.) * 60))))); \
}

#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE
} // namespace