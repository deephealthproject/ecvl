/*
* ECVL - European Computer Vision Library
* Version: 0.3.4
* copyright (c) 2021, Universit� degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#ifndef ECVL_FPGA_HAL_H_
#define ECVL_FPGA_HAL_H_

#include "ecvl/core/hal.h"
#include "ecvl/core/image.h"
#include <iostream>
//#include "../../../../../3rdparty/xfopencv/examples_sdaccel/resize/xcl2.hpp"

extern cl::Context *context;
extern cl::CommandQueue *q;

extern cl::Kernel kernel_otsu_threshold, kernel_threshold, kernel_mirror2d, kernel_flip2d, kernel_rgb_2_gray, kernel_gaussian_blur, kernel_resize, kernel_warp_transform;
extern cl::Kernel kernel_integral_image, kernel_histogram, kernel_gamma_correction, kernel_filter2d, kernel_min_max, kernel_gray_2_rgb;

namespace ecvl
{
/** @brief FPGA specific Hardware Abstraction Layer

*/

void fpga_init();
//void ReturnBuffer(cv::Mat& src,  ecvl::Image& dst);

class FpgaHal : public HardwareAbstractionLayer
{
public:
    uint8_t* MemAllocate(size_t nbytes) override
    {
        printf("FPGA: MemAllocate\n");
        cl::Buffer *imageToDevice = new cl::Buffer(*context, CL_MEM_READ_ONLY, nbytes);
        return (uint8_t*) imageToDevice;
    }
    void MemDeallocate(uint8_t* data) override
    {
        printf("FPGA: MemDeallocate\n");
        delete[] data;
    }
    uint8_t* MemCopy(uint8_t* dst, const uint8_t* src, size_t nbytes) override
    {
        printf("FPGA: MemCopy %zu bytes\n", nbytes);
        cl::Buffer *buffer_a = (cl::Buffer*) dst;
        return (uint8_t*) (*q).enqueueWriteBuffer(*buffer_a, CL_TRUE, 0, nbytes, src);
    }

    int fpga_initialized = 0;

    static FpgaHal* GetInstance();

    void FromCpu(Image& src) override;
    void ToCpu(Image& src) override;


    void CopyImage(const Image& src, Image& dst) override;
    void RearrangeChannels(const Image& src, Image& dst, const std::vector<int>& bindings) override;
    void ConvertTo(const Image& src, Image& dst, DataType dtype, bool saturate) override;

    void ResizeDim(const ecvl::Image& src, ecvl::Image& dst, const std::vector<int>& newdims, InterpolationType interp) override;
    void ResizeScale(const Image& src, Image& dst, const std::vector<double>& scales, InterpolationType interp) override;
    void Flip2D(const ecvl::Image& src, ecvl::Image& dst) override;
    void Mirror2D(const ecvl::Image& src, ecvl::Image& dst) override;
    void Rotate2D(const ecvl::Image& src, ecvl::Image& dst, double angle, const std::vector<double>& center, double scale, InterpolationType interp) override;
    void RotateFullImage2D(const ecvl::Image& src, ecvl::Image& dst, double angle, double scale, InterpolationType interp) override;
    void ChangeColorSpace(const Image& src, Image& dst, ColorType new_type) override;
    void Threshold(const Image& src, Image& dst, double thresh, double maxval, ThresholdingType thresh_type) override;
    std::vector<double> Histogram(const Image& src) override;
    int OtsuThreshold(const Image& src) override;
    void Filter2D(const Image& src, Image& dst, const Image& ker, DataType type) override;
    void SeparableFilter2D(const Image& src, Image& dst, const std::vector<double>& kerX, const std::vector<double>& kerY, DataType type) override;
    void GaussianBlur(const Image& src, Image& dst, int sizeX, int sizeY, double sigmaX, double sigmaY) override;
    void AdditiveLaplaceNoise(const Image& src, Image& dst, double std_dev) override;
    void AdditivePoissonNoise(const Image& src, Image& dst, double lambda) override;
    void GammaContrast(const Image& src, Image& dst, double gamma) override;
    void CoarseDropout(const Image& src, Image& dst, double p, double drop_size, bool per_channel) override;
    void IntegralImage(const Image& src, Image& dst, DataType dst_type) override;
    void NonMaximaSuppression(const Image& src, Image& dst) override;
    std::vector<ecvl::Point2i> GetMaxN(const Image& src, size_t n) override;
    int ConnectedComponentsLabeling(const Image& src, Image& dst) override;
    void FindContours(const Image& src, std::vector<std::vector<ecvl::Point2i>>& contours) override;
    void Stack(const std::vector<Image>& src, Image& dst) override;
    void HConcat(const std::vector<Image>& src, Image& dst) override;
    void VConcat(const std::vector<Image>& src, Image& dst) override;
    void Morphology(const Image& src, Image& dst, MorphType op, Image& kernel, Point2i anchor, int iterations, BorderType border_type, const int& border_value) override;
    void Inpaint(const Image& src, Image& dst, const Image& inpaintMask, double inpaintRadius, InpaintType flag) override;
    void MeanStdDev(const Image& src, std::vector<double>& mean, std::vector<double>& stddev) override;
    void Transpose(const Image& src, Image& dst) override;
    void GridDistortion(const Image& src, Image& dst, int num_steps, const std::array<float, 2>& distort_limit, InterpolationType interp, BorderType border_type, const int& border_value, const unsigned seed) override;
    void ElasticTransform(const Image& src, Image& dst, double alpha, double sigma, InterpolationType interp, BorderType border_type, const int& border_value, const unsigned seed) override;
    void OpticalDistortion(const Image& src, Image& dst, const std::array<float, 2>& distort_limit, const std::array<float, 2>& shift_limit, InterpolationType interp, BorderType border_type, const int& border_value, const unsigned seed) override;
    void Salt(const Image& src, Image& dst, double p, bool per_channel, const unsigned seed) override;
    void Pepper(const Image& src, Image& dst, double p, bool per_channel, const unsigned seed) override;
    void SaltAndPepper(const Image& src, Image& dst, double p, bool per_channel, const unsigned seed) override;
  
    void SliceTimingCorrection(const Image& src, Image& dst, bool odd, bool down) override;
    
    // void Moments(const Image& src, Image& moments, int order, DataType type) override;
    void CentralMoments(const Image& src, Image& moments, std::vector<double> center, int order, DataType type) override;
    void DrawEllipse(Image& src, ecvl::Point2i center, ecvl::Size2i axes, double angle, const ecvl::Scalar& color, int thickness) override;
    std::vector<int> OtsuMultiThreshold(const Image& src, int n_thresholds) override;
    void MultiThreshold(const Image& src, Image& dst, const std::vector<int>& thresholds, int minval, int maxval) override;
    void Normalize(const Image& src, Image& dst, const double& mean, const double& std) override;
    void Normalize(const Image& src, Image& dst, const std::vector<double>& mean, const std::vector<double>& std) override;
    void CenterCrop(const Image& src, Image& dst, const std::vector<int>& size) override;
    void ScaleTo(const Image& src, Image& dst, const double& new_min, const double& new_max) override;
    void Pad(const Image& src, Image& dst, const std::vector<int>& padding, BorderType border_type, const int& border_value) override;
    void RandomCrop(const Image& src, Image& dst, const std::vector<int>& size, bool pad_if_needed, BorderType border_type, const int& border_value, const unsigned seed) override;

    void Neg(const Image& src, Image& dst, DataType dst_type, bool saturate) override;
    void Add(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override;
    void Sub(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override;
    void Mul(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override;
    void Div(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override;

#define ECVL_TUPLE(name, size, type, ...) \
    void Add(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) override; \
    void Add(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override; \
                                                                                                   \
    void Sub(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) override; \
    void Sub(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override; \
                                                                                                   \
    void Mul(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) override; \
    void Mul(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override; \
                                                                                                   \
    void Div(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) override; \
    void Div(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override; \
                                                                                                   \
    void SetTo(Image& src, type value) override;

#include "datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE

};

} // namespace ecvl
#endif // ECVL_FPGA_HAL_H_
