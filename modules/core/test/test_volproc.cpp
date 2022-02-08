/*
* ECVL - European Computer Vision Library
* Version: 1.0.1
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
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
    Image g1_spacing_##type = Image({ 1, 1, 1, 1 }, DataType::type, "xyzt", ColorType::GRAY, {1, 2, 3, 4}); \
    View<DataType::type> g1_spacing_##type##_v; \
    Image g2_##type = Image({ 2, 2, 1 }, DataType::type, "xyc", ColorType::GRAY); \
    View<DataType::type> g2_##type##_v; \
    Image rgb2_##type = Image({ 2, 2, 3 }, DataType::type, "xyc", ColorType::RGB); \
    View<DataType::type> rgb2_##type##_v; \
    Image rgb2_spacing_##type = Image({ 2, 2, 3, 1 }, DataType::type, "xyzt", ColorType::RGB, {2, 4, 7, 1}); \
    View<DataType::type> rgb2_spacing_##type##_v;

#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE

    void SetUp() override
    {
        out = Image();

#define ECVL_TUPLE(type, ...) \
        g1_##type##_v = g1_##type; \
        g1_##type##_v({ 0,0,0 }) = 50; \
        \
        g1_spacing_##type##_v = g1_spacing_##type; \
        g1_spacing_##type##_v({ 0,0,0,0 }) = 50; \
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
        \
        rgb2_spacing_##type##_v = rgb2_spacing_##type; \
        rgb2_spacing_##type##_v({ 0,0,0,0 }) = 50; rgb2_spacing_##type##_v({ 1,0,0,0 }) = 32; \
        rgb2_spacing_##type##_v({ 0,1,0,0 }) = 14; rgb2_spacing_##type##_v({ 1,1,0,0 }) = 60; \
        rgb2_spacing_##type##_v({ 0,0,1,0 }) = 50; rgb2_spacing_##type##_v({ 1,0,1,0 }) = 32; \
        rgb2_spacing_##type##_v({ 0,1,1,0 }) = 14; rgb2_spacing_##type##_v({ 1,1,1,0 }) = 60; \
        rgb2_spacing_##type##_v({ 0,0,2,0 }) = 50; rgb2_spacing_##type##_v({ 1,0,2,0 }) = 32; \
        rgb2_spacing_##type##_v({ 0,1,2,0 }) = 14; rgb2_spacing_##type##_v({ 1,1,2,0 }) = 60;

#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE
    }
};

using VolProc = TCore;

#define ECVL_TUPLE(type, ...) \
TEST_F(VolProc, SliceTimingCorrection##type) \
{ \
    EXPECT_THROW(SliceTimingCorrection(g1_##type, out), std::runtime_error); \
    \
    SliceTimingCorrection(g1_spacing_##type, out); \
    View<DataType::float32> out_v(out); \
    EXPECT_TRUE(out_v({ 0,0,0,0 }) == 50); \
    EXPECT_THAT(out.dims_, testing::ElementsAre(1, 1, 1, 1)); \
    \
    SliceTimingCorrection(rgb2_spacing_##type, out); \
    out_v = out; \
    EXPECT_TRUE(out_v({ 0,0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0,0 }) == 32); \
    EXPECT_TRUE(out_v({ 0,0,1,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,1,0 }) == 32); \
    EXPECT_TRUE(out_v({ 0,0,2,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,2,0 }) == 32); \
    EXPECT_THAT(out.dims_, testing::ElementsAre(2, 2, 3, 1)); \
} \
\
TEST_F(VolProc, SliceTimingCorrectionSameDst##type) \
{ \
    EXPECT_THROW(SliceTimingCorrection(g1_##type, g1_##type), std::runtime_error); \
    \
    SliceTimingCorrection(g1_spacing_##type, g1_spacing_##type); \
    View<DataType::float32> out_v(g1_spacing_##type); \
    EXPECT_TRUE(out_v({ 0,0,0,0 }) == 50); \
    EXPECT_THAT(g1_spacing_##type.dims_, testing::ElementsAre(1, 1, 1, 1)); \
    \
    SliceTimingCorrection(rgb2_spacing_##type, rgb2_spacing_##type); \
    out_v = rgb2_spacing_##type; \
    EXPECT_TRUE(out_v({ 0,0,0,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,0,0 }) == 32); \
    EXPECT_TRUE(out_v({ 0,0,1,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,1,0 }) == 32); \
    EXPECT_TRUE(out_v({ 0,0,2,0 }) == 50); EXPECT_TRUE(out_v({ 1,0,2,0 }) == 32); \
    EXPECT_THAT(rgb2_spacing_##type.dims_, testing::ElementsAre(2, 2, 3, 1)); \
}

#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE
} // namespace