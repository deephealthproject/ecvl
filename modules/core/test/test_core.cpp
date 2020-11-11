/*
* ECVL - European Computer Vision Library
* Version: 0.2.1
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

#ifdef ECVL_GPU
#include <cuda_runtime.h>
#include "test_cuda.h"
#include "ecvl/core/cuda/common.h"
#endif 

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
        g2_##type##_v({ 0,0,0 }) = 50; g2_##type##_v({ 1,0,0 }) = 32; \
        g2_##type##_v({ 0,1,0 }) = 14; g2_##type##_v({ 1,1,0 }) = 60;

#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE
    }
};

using CoreImage = TCore;
using CoreArithmetics = TCore;

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
    EXPECT_TRUE(img_v({ 0,0,0 }) == 50); EXPECT_TRUE(img_v({ 1,0,0 }) == 32); \
    EXPECT_TRUE(img_v({ 0,1,0 }) == 14); EXPECT_TRUE(img_v({ 1,1,0 }) == 60); \
} \
\
TEST_F(CoreImage, Copy##type) \
{ \
    Image img(g2_##type); \
    View<DataType::type> img_v(img); \
    EXPECT_TRUE(img_v({ 0,0,0 }) == 50); EXPECT_TRUE(img_v({ 1,0,0 }) == 32); \
    EXPECT_TRUE(img_v({ 0,1,0 }) == 14); EXPECT_TRUE(img_v({ 1,1,0 }) == 60); \
} \
\
TEST_F(CoreImage, CopyNonContiguousXYZC##type) \
{ \
    Image src({ 2, 3, 3, 1 }, DataType::type, "xyzc", ColorType::GRAY); \
    View<DataType::type> src_v(src, { 0, 0, 0, 0 }, { 2, 2, 2, 1 }); \
    src_v({ 0,0,0,0 }) = 50; src_v({ 0,0,1,0 }) = 32; \
    src_v({ 0,1,0,0 }) = 14; src_v({ 0,1,1,0 }) = 60; \
    src_v({ 1,0,0,0 }) = 54; src_v({ 1,0,1,0 }) = 41; \
    src_v({ 1,1,0,0 }) = 97; src_v({ 1,1,1,0 }) = 79; \
    Image dst(src_v); \
    View<DataType::type> dst_v(dst); \
    EXPECT_TRUE(dst_v({ 0,0,0,0 }) == 50); EXPECT_TRUE(dst_v({ 0,0,1,0 }) == 32); \
    EXPECT_TRUE(dst_v({ 0,1,0,0 }) == 14); EXPECT_TRUE(dst_v({ 0,1,1,0 }) == 60); \
    EXPECT_TRUE(dst_v({ 1,0,0,0 }) == 54); EXPECT_TRUE(dst_v({ 1,0,1,0 }) == 41); \
    EXPECT_TRUE(dst_v({ 1,1,0,0 }) == 97); EXPECT_TRUE(dst_v({ 1,1,1,0 }) == 79); \
} \
\
TEST_F(CoreImage, RearrangeXYZC##type) \
{ \
    Image img({ 3, 4, 3, 2 }, DataType::type, "cxyz", ColorType::RGB); \
    View<DataType::type> view(img); \
    auto it = view.Begin(); \
    for (uint8_t i = 0; i < 24 * 3; ++i) { \
        *reinterpret_cast<TypeInfo_t<DataType::type>*>(it.ptr_) = i; \
        ++it; \
    } \
    Image img2; \
    RearrangeChannels(img, img2, "xyzc"); \
    View<DataType::type> view2(img2); \
    \
    EXPECT_TRUE(view2({ 2, 0, 1, 0 }) == 42); \
    EXPECT_TRUE(view2({ 3, 1, 1, 2 }) == 59); \
    EXPECT_TRUE(view2({ 0, 2, 0, 1 }) == 25); \
    EXPECT_TRUE(view2({ 1, 2, 0, 1 }) == 28); \
} \
TEST_F(CoreImage, CopyNonContiguousXYO##type) \
{ \
    Image src({ 2, 3, 3 }, DataType::type, "xyo", ColorType::GRAY); \
    View<DataType::type> src_v(src, { 0, 0, 0 }, { 2, 2, 2 }); \
    src_v({ 0,0,0 }) = 50; src_v({ 0,0,1 }) = 32; \
    src_v({ 0,1,0 }) = 14; src_v({ 0,1,1 }) = 60; \
    src_v({ 1,0,0 }) = 54; src_v({ 1,0,1 }) = 41; \
    src_v({ 1,1,0 }) = 97; src_v({ 1,1,1 }) = 79; \
    Image dst(src_v); \
    View<DataType::type> dst_v(dst); \
    EXPECT_TRUE(dst_v({ 0,0,0 }) == 50); EXPECT_TRUE(dst_v({ 0,0,1 }) == 32); \
    EXPECT_TRUE(dst_v({ 0,1,0 }) == 14); EXPECT_TRUE(dst_v({ 0,1,1 }) == 60); \
    EXPECT_TRUE(dst_v({ 1,0,0 }) == 54); EXPECT_TRUE(dst_v({ 1,0,1 }) == 41); \
    EXPECT_TRUE(dst_v({ 1,1,0 }) == 97); EXPECT_TRUE(dst_v({ 1,1,1 }) == 79); \
} \
\
TEST_F(CoreImage, RearrangeXYO##type) \
{ \
    Image img({ 6, 4, 3 }, DataType::type, "oxy", ColorType::RGB); \
    View<DataType::type> view(img); \
    auto it = view.Begin(); \
    for (uint8_t i = 0; i < 24 * 3; ++i) { \
        *reinterpret_cast<TypeInfo_t<DataType::type>*>(it.ptr_) = i; \
        ++it; \
    } \
    Image img2; \
    RearrangeChannels(img, img2, "xyo"); \
    View<DataType::type> view2(img2); \
    \
    EXPECT_TRUE(view2({ 3, 1, 0 }) == 42); \
    EXPECT_TRUE(view2({ 1, 2, 5 }) == 59); \
    EXPECT_TRUE(view2({ 0, 1, 1 }) == 25); \
    EXPECT_TRUE(view2({ 0, 1, 4 }) == 28); \
} \
\
TEST_F(CoreImage, ToFpga##type) \
{ \
    Image tmp(g2_##type); \
    EXPECT_THROW(tmp.To(Device::FPGA), std::runtime_error); \
} \
\
\
\
TEST_F(CoreArithmetics, AddScalar##type) \
{ \
    g2_##type.Add(10); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 60); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 42); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 24); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 70); \
    g2_##type.Add(10, false); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 70); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 52); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 34); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 80); \
} \
\
TEST_F(CoreArithmetics, AddImage##type) \
{ \
    g2_##type.Add(g2_##type); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 100); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 64); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 28); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 120); \
} \
\
TEST_F(CoreArithmetics, AddImageNonSaturate##type) \
{ \
    g2_##type.Add(g2_##type, false); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 100); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 64); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 28); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 120); \
} \
\
TEST_F(CoreArithmetics, SubScalar##type) \
{ \
    g2_##type.Sub(10); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 40); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 22); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 4); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 50); \
} \
\
TEST_F(CoreArithmetics, SubScalarNonSaturate##type) \
{ \
    g2_##type.Sub(10, false); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 40); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 22); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 4); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 50); \
} \
\
TEST_F(CoreArithmetics, SubImage##type) \
{ \
    g2_##type.Sub(g2_##type); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 0); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 0); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 0); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 0); \
} \
\
TEST_F(CoreArithmetics, MulScalar##type) \
{ \
    g2_##type.Mul(2); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 100); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 64); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 28); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 120); \
} \
\
TEST_F(CoreArithmetics, MulScalarNonSaturate##type) \
{ \
    g2_##type.Mul(2, false); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 100); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 64); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 28); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 120); \
} \
\
TEST_F(CoreArithmetics, MulImage##type) \
{ \
    Image tmp(g2_##type); \
    View<DataType::type> tmp_v(tmp); \
    auto i = tmp.Begin<uint8_t>(), e = tmp.End<uint8_t>(); \
    for (; i != e; ++i) { \
        *reinterpret_cast<TypeInfo_t<DataType::type>*>(i.ptr_) = 2; \
    } \
    g2_##type.Mul(tmp); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 100); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 64); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 28); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 120); \
} \
\
TEST_F(CoreArithmetics, DivScalar##type) \
{ \
    g2_##type.Div(2); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 25); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 16); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 7); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 30); \
} \
\
TEST_F(CoreArithmetics, DivScalarNonSaturate##type) \
{ \
    g2_##type.Div(2, false); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 25); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 16); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 7); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 30); \
} \
\
TEST_F(CoreArithmetics, DivImage##type) \
{ \
    Image tmp(g2_##type); \
    View<DataType::type> tmp_v(tmp); \
    auto i = tmp.Begin<uint8_t>(), e = tmp.End<uint8_t>(); \
    for (; i != e; ++i) { \
        *reinterpret_cast<TypeInfo_t<DataType::type>*>(i.ptr_) = 2; \
    } \
    g2_##type.Div(tmp); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == 25); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == 16); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == 7); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == 30); \
}\
\
TEST_F(CoreArithmetics, SetTo##type) \
{ \
    g2_##type.SetTo(0); \
    View<DataType::type> my_view(g2_##type); \
    auto i = my_view.Begin(), e = my_view.End(); \
    for (; i != e; ++i) { \
        EXPECT_TRUE(*i == static_cast<TypeInfo_t<DataType::type>>(0)); \
    } \
} \

#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE

#define ECVL_TUPLE(type, ...) \
TEST_F(CoreArithmetics, Neg##type) \
{ \
    g2_##type.Neg(); \
    EXPECT_TRUE(g2_##type##_v({ 0,0,0 }) == -50); EXPECT_TRUE(g2_##type##_v({ 1,0,0 }) == -32); \
    EXPECT_TRUE(g2_##type##_v({ 0,1,0 }) == -14); EXPECT_TRUE(g2_##type##_v({ 1,1,0 }) == -60); \
}

#include "ecvl/core/datatype_existing_tuples_signed.inc.h"
#undef ECVL_TUPLE

#ifdef ECVL_GPU
#define ECVL_TUPLE(type, ...) \
TEST_F(CoreImage, CpuToGpu##type) \
{ \
    Image tmp(g2_##type); \
    tmp.To(Device::GPU); \
    uint8_t res_h; \
    uint8_t* res_d; \
    checkCudaError(cudaMalloc(&res_d, 1)); \
    RunTestCpuToGpuKernel##type(tmp.data_, res_d); \
    checkCudaError(cudaMemcpy(&res_h, res_d, 1, cudaMemcpyDeviceToHost)); \
    checkCudaError(cudaFree(res_d)); \
    EXPECT_TRUE(res_h == 1); \
} \
\
TEST_F(CoreImage, GpuToCpu##type) { \
    Image gpu_img({ 2, 2, 1 }, DataType::type, "xyc", ColorType::GRAY, { 1, 1, 1 }, Device::GPU); \
    RunTestGpuToCpuKernel##type(gpu_img.data_); \
    checkCudaError(cudaDeviceSynchronize()); \
    gpu_img.To(Device::CPU); \
    View<DataType::type> img_v(gpu_img); \
    EXPECT_TRUE(img_v({ 0,0,0 }) == 50); EXPECT_TRUE(img_v({ 1,0,0 }) == 32); \
    EXPECT_TRUE(img_v({ 0,1,0 }) == 14); EXPECT_TRUE(img_v({ 1,1,0 }) == 60); \
}
#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE
#endif

#if 0 // Functions reimplementation needed
TEST_F(CoreArithmetics, Anduint8)
{
    Image tmp(g2_uint8);
    View<DataType::uint8> tmp_v(tmp);
    tmp_v({ 0,0,0 }) = 49; tmp_v({ 1,0,0 }) = 31;
    tmp_v({ 0,1,0 }) = 13; tmp_v({ 1,1,0 }) = 59;
    And(g2_uint8, tmp, out);
    View<DataType::uint8> out_v(out);
    EXPECT_TRUE(out_v({ 0,0,0 }) == 48); EXPECT_TRUE(out_v({ 1,0,0 }) == 0);
    EXPECT_TRUE(out_v({ 0,1,0 }) == 12); EXPECT_TRUE(out_v({ 1,1,0 }) == 56);
}

TEST_F(CoreArithmetics, Oruint8)
{
    Image tmp(g2_uint8);
    View<DataType::uint8> tmp_v(tmp);
    tmp_v({ 0,0,0 }) = 49; tmp_v({ 1,0,0 }) = 31;
    tmp_v({ 0,1,0 }) = 13; tmp_v({ 1,1,0 }) = 59;
    Or(g2_uint8, tmp, out);
    View<DataType::uint8> out_v(out);
    EXPECT_TRUE(out_v({ 0,0,0 }) == 51); EXPECT_TRUE(out_v({ 1,0,0 }) == 63);
    EXPECT_TRUE(out_v({ 0,1,0 }) == 15); EXPECT_TRUE(out_v({ 1,1,0 }) == 63);
}
#endif // 0
}