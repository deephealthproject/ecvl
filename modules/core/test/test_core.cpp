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

namespace
{
class TCore : public ::testing::Test
{
protected:
    Image out;

#define ECVL_TUPLE(type, ...) \
    Image g2_##type = Image({ 2, 2, 1 }, DataType::type, "xyc", ColorType::GRAY); \
    View<DataType::type> g2_##type##_v;
    //Image g3_##type = Image({ 3, 3, 1 }, DataType::type, "xyc", ColorType::GRAY); \
    //View<DataType::type> g3_##type##_v; \

#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE

    void SetUp() override
    {
#define ECVL_TUPLE(type, ...) \
        g2_##type##_v = g2_##type; \
        g2_##type##_v({ 0,0,0 }) = 50; g2_##type##_v({ 1,0,0 }) = 33; \
        g2_##type##_v({ 0,1,0 }) = 15; g2_##type##_v({ 1,1,0 }) = 92;

#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE
    }
};

using CoreImage = TCore;

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

#define ECVL_TUPLE(type, ...) \
TEST_F(CoreImage, Move##type) \
{ \
    Image img(std::move(g2_##type)); \
    View<DataType::type> img_v(img); \
    EXPECT_NEAR(img_v({ 0,0,0 }), 50, 0.1); EXPECT_NEAR(img_v({ 1,0,0 }), 33, 0.1); \
    EXPECT_NEAR(img_v({ 0,1,0 }), 15, 0.1); EXPECT_NEAR(img_v({ 1,1,0 }), 92, 0.1); \
}

#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE

#define ECVL_TUPLE(type, ...) \
TEST_F(CoreImage, Copy##type) \
{ \
    Image img(g2_##type); \
    View<DataType::type> img_v(img); \
    EXPECT_NEAR(img_v({ 0,0,0 }), 50, 0.1); EXPECT_NEAR(img_v({ 1,0,0 }), 33, 0.1); \
    EXPECT_NEAR(img_v({ 0,1,0 }), 15, 0.1); EXPECT_NEAR(img_v({ 1,1,0 }), 92, 0.1); \
}

#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE

#define ECVL_TUPLE(type, ...) \
TEST_F(CoreImage, Neg##type) \
{ \
    Neg(g2_##type); \
    EXPECT_NEAR(g2_##type##_v({ 0,0,0 }), -50, 0.1); EXPECT_NEAR(g2_##type##_v({ 1,0,0 }), -33, 0.1); \
    EXPECT_NEAR(g2_##type##_v({ 0,1,0 }), -15, 0.1); EXPECT_NEAR(g2_##type##_v({ 1,1,0 }), -92, 0.1); \
}
#include "ecvl/core/datatype_existing_tuples_signed.inc.h"
#undef ECVL_TUPLE

#define ECVL_TUPLE(type, ...) \
TEST_F(CoreImage, Rearrange##type) \
{ \
    Image img({ 3, 4, 3, 2 }, DataType::type, "cxyz", ColorType::RGB); \
    View<DataType::type> view(img); \
    auto it = view.Begin(); \
    for (uint8_t i = 0; i < 24 * 3; ++i) { \
        *it = i; \
        ++it; \
    } \
    Image img2; \
    RearrangeChannels(img, img2, "xyzc"); \
    View<DataType::type> view2(img2); \
    \
    EXPECT_EQ(view2({ 2, 0, 1, 0 }), 42); \
    EXPECT_EQ(view2({ 3, 1, 1, 2 }), 59); \
    EXPECT_EQ(view2({ 0, 2, 0, 1 }), 25); \
    EXPECT_EQ(view2({ 1, 2, 0, 1 }), 28); \
}

#include "ecvl/core/datatype_existing_tuples_signed.inc.h"
#undef ECVL_TUPLE
}