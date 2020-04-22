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
//using CoreArithmetics = TCore;

#define ECVL_TUPLE(type, ...) \
TEST_F(Imgproc, Threshold##type) \
{ \
    Threshold(g2_##type, out, 35, 127, ThresholdingType::BINARY); \
    View<DataType::type> out_v(out); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 127); EXPECT_TRUE(out_v({ 1,0,0 }) == 0); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 0); EXPECT_TRUE(out_v({ 1,1,0 }) == 127); \
    Threshold(g2_##type, out, 35, 127, ThresholdingType::BINARY_INV); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 0); EXPECT_TRUE(out_v({ 1,0,0 }) == 127); \
    EXPECT_TRUE(out_v({ 0,1,0 }) == 127); EXPECT_TRUE(out_v({ 1,1,0 }) == 0); \
} \
\
TEST_F(Imgproc, ThresholdSameDst##type) \
{ \
    Threshold(g2_##type, g2_##type, 35, 127, ThresholdingType::BINARY); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 127); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 0); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 0); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 127); \
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
    EXPECT_TRUE(g1_##type##_v({ 0,0,0 }) == 50); \
    \
    Mirror2D(g2_##type, g2_##type); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 32); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 50); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 60); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 14); \
    \
    Mirror2D(rgb2_##type, rgb2_##type); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,0,0 }) == 32); EXPECT_TRUE(rgb2_##type##_v({ 1,0,0 }) == 50); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,1,0 }) == 60); EXPECT_TRUE(rgb2_##type##_v({ 1,1,0 }) == 14); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,0,1 }) == 32); EXPECT_TRUE(rgb2_##type##_v({ 1,0,1 }) == 50); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,1,1 }) == 60); EXPECT_TRUE(rgb2_##type##_v({ 1,1,1 }) == 14); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,0,2 }) == 32); EXPECT_TRUE(rgb2_##type##_v({ 1,0,2 }) == 50); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,1,2 }) == 60); EXPECT_TRUE(rgb2_##type##_v({ 1,1,2 }) == 14); \
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
    EXPECT_TRUE(g1_##type##_v({ 0,0,0 }) == 50); \
    \
    Flip2D(g2_##type, g2_##type); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 14); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 60); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 50); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 32); \
    \
    Flip2D(rgb2_##type, rgb2_##type); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 14); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 60); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 50); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 32); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,1 }) == 14); EXPECT_TRUE(g2_##type##_v({ 1,0,1 }) == 60); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,1 }) == 50); EXPECT_TRUE(g2_##type##_v({ 1,1,1 }) == 32); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,2 }) == 14); EXPECT_TRUE(g2_##type##_v({ 1,0,2 }) == 60); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,2 }) == 50); EXPECT_TRUE(g2_##type##_v({ 1,1,2 }) == 32); \
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
    EXPECT_TRUE(g1_##type##_v({ 0,0,0 }) == 50); EXPECT_TRUE(g1_##type##_v({ 1,0,0 }) == 50); \
    \
    HConcat({ g2_##type, g2_##type }, g2_##type); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 50); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) = 32); EXPECT_TRUE(g2_##type##_v({ 2,0,0 }) == 50); EXPECT_TRUE(g2_##type##_v({ 3,0,0 }) = 32); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 14); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) = 60); EXPECT_TRUE(g2_##type##_v({ 2,1,0 }) == 14); EXPECT_TRUE(g2_##type##_v({ 3,1,0 }) = 60); \
    \
    HConcat({ rgb2_##type, rgb2_##type }, rgb2_##type); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,0,0 }) == 50); EXPECT_TRUE(rgb2_##type##_v({ 1,0,0 }) = 32); EXPECT_TRUE(rgb2_##type##_v({ 2,0,0 }) == 50); EXPECT_TRUE(rgb2_##type##_v({ 3,0,0 }) = 32); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,1,0 }) == 14); EXPECT_TRUE(rgb2_##type##_v({ 1,1,0 }) = 60); EXPECT_TRUE(rgb2_##type##_v({ 2,1,0 }) == 14); EXPECT_TRUE(rgb2_##type##_v({ 3,1,0 }) = 60); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,0,1 }) == 50); EXPECT_TRUE(rgb2_##type##_v({ 1,0,1 }) = 32); EXPECT_TRUE(rgb2_##type##_v({ 2,0,1 }) == 50); EXPECT_TRUE(rgb2_##type##_v({ 3,0,1 }) = 32); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,1,1 }) == 14); EXPECT_TRUE(rgb2_##type##_v({ 1,1,1 }) = 60); EXPECT_TRUE(rgb2_##type##_v({ 2,1,1 }) == 14); EXPECT_TRUE(rgb2_##type##_v({ 3,1,1 }) = 60); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,0,2 }) == 50); EXPECT_TRUE(rgb2_##type##_v({ 1,0,2 }) = 32); EXPECT_TRUE(rgb2_##type##_v({ 2,0,2 }) == 50); EXPECT_TRUE(rgb2_##type##_v({ 3,0,2 }) = 32); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,1,2 }) == 14); EXPECT_TRUE(rgb2_##type##_v({ 1,1,2 }) = 60); EXPECT_TRUE(rgb2_##type##_v({ 2,1,2 }) == 14); EXPECT_TRUE(rgb2_##type##_v({ 3,1,2 }) = 60); \
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
    VConcat({ g1_##type, g1_##type },  g1_##type); \
    EXPECT_TRUE(g1_##type##_v({ 0,0,0 }) == 50); \
    EXPECT_TRUE(g1_##type##_v({ 0,1,0 }) == 50); \
    \
    VConcat({ g2_##type, g2_##type }, g2_##type); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 50); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) = 32); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 14); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) = 60); \
    EXPECT_TRUE(g2_##type##_v({ 0,2,0 }) == 50); EXPECT_TRUE(g2_##type##_v({ 1,2,0 }) = 32); \
    EXPECT_TRUE(g2_##type##_v({ 0,3,0 }) == 14); EXPECT_TRUE(g2_##type##_v({ 1,3,0 }) = 60); \
    \
    VConcat({ rgb2_##type, rgb2_##type },  rgb2_##type); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,0,0 }) == 50); EXPECT_TRUE(rgb2_##type##_v({ 1,0,0 }) = 32); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,1,0 }) == 14); EXPECT_TRUE(rgb2_##type##_v({ 1,1,0 }) = 60); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,2,0 }) == 50); EXPECT_TRUE(rgb2_##type##_v({ 1,2,0 }) = 32); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,3,0 }) == 14); EXPECT_TRUE(rgb2_##type##_v({ 1,3,0 }) = 60); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,0,1 }) == 50); EXPECT_TRUE(rgb2_##type##_v({ 1,0,1 }) = 32); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,1,1 }) == 14); EXPECT_TRUE(rgb2_##type##_v({ 1,1,1 }) = 60); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,2,1 }) == 50); EXPECT_TRUE(rgb2_##type##_v({ 1,2,1 }) = 32); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,3,1 }) == 14); EXPECT_TRUE(rgb2_##type##_v({ 1,3,1 }) = 60); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,0,2 }) == 50); EXPECT_TRUE(rgb2_##type##_v({ 1,0,2 }) = 32); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,1,2 }) == 14); EXPECT_TRUE(rgb2_##type##_v({ 1,1,2 }) = 60); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,2,2 }) == 50); EXPECT_TRUE(rgb2_##type##_v({ 1,2,2 }) = 32); \
    EXPECT_TRUE(rgb2_##type##_v({ 0,3,2 }) == 14); EXPECT_TRUE(rgb2_##type##_v({ 1,3,2 }) = 60); \
} \
\
TEST_F(Imgproc, ResizeDim##type) \
{ \
    ResizeDim(g2_uint8, out, { 1, 1 }, InterpolationType::nearest); \
    View<DataType::uint8> out_v(out); \
    EXPECT_THAT(out.dims_, testing::ElementsAre(1, 1, 1)); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); \
    \
    ResizeDim(g2_uint8, out, { 2, 1 }, InterpolationType::nearest); \
    out_v = out; \
    EXPECT_THAT(out.dims_, testing::ElementsAre(2, 1, 1)); \
    EXPECT_TRUE(out_v({ 0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0 }) == 32); \
}

#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE

}