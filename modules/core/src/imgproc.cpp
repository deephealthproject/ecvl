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

#include "ecvl/core/imgproc.h"

#include <opencv2/imgproc.hpp>

namespace ecvl
{
using namespace std;

void AlwaysCheck(const ecvl::Image& src, const ecvl::Image& dst)
{
    if (src.IsEmpty()) {
        ECVL_ERROR_EMPTY_IMAGE
    }

    if (src.dev_ != dst.dev_ && dst.dev_ != Device::NONE) {
        ECVL_ERROR_DIFFERENT_DEVICES
    }
}

int GetOpenCVInterpolation(InterpolationType interp)
{
    switch (interp) {
    case InterpolationType::nearest:    return cv::INTER_NEAREST;
    case InterpolationType::linear:     return cv::INTER_LINEAR;
    case InterpolationType::area:       return cv::INTER_AREA;
    case InterpolationType::cubic:      return cv::INTER_CUBIC;
    case InterpolationType::lanczos4:   return cv::INTER_LANCZOS4;
    default:
        // This error occurs when interp is an uninitialized variable
        ECVL_ERROR_NOT_REACHABLE_CODE
    }
}

void ResizeDim(const ecvl::Image& src, ecvl::Image& dst, const std::vector<int>& newdims, InterpolationType interp)
{
    AlwaysCheck(src, dst);

    if (newdims.size() != 2) {
        ECVL_ERROR_WRONG_PARAMS("Number of dimensions specified doesn't match image dimensions")
    }

    src.hal_->ResizeDim(src, dst, newdims, interp);
}

void ResizeScale(const Image& src, Image& dst, const std::vector<double>& scales, InterpolationType interp)
{
    AlwaysCheck(src, dst);

    if (scales.size() != 2) {
        ECVL_ERROR_WRONG_PARAMS("Number of dimensions specified doesn't match image dimensions")
    }

    src.hal_->ResizeScale(src, dst, scales, interp);
}

void Flip2D(const ecvl::Image& src, ecvl::Image& dst)
{
    AlwaysCheck(src, dst);

    src.hal_->Flip2D(src, dst);
}

void Mirror2D(const ecvl::Image& src, ecvl::Image& dst)
{
    AlwaysCheck(src, dst);

    src.hal_->Mirror2D(src, dst);
}

void Rotate2D(const ecvl::Image& src, ecvl::Image& dst, double angle, const std::vector<double>& center, double scale, InterpolationType interp)
{
    AlwaysCheck(src, dst);

    if (!center.empty() && center.size() != 2) {
        ECVL_ERROR_WRONG_PARAMS("Rotation center must have two dimensions")
    }

    src.hal_->Rotate2D(src, dst, angle, center, scale, interp);
}

void RotateFullImage2D(const ecvl::Image& src, ecvl::Image& dst, double angle, double scale, InterpolationType interp)
{
    AlwaysCheck(src, dst);

    src.hal_->RotateFullImage2D(src, dst, angle, scale, interp);
}

void ChangeColorSpace(const Image& src, Image& dst, ColorType new_type)
{
    AlwaysCheck(src, dst);

    if (src.colortype_ == ColorType::none || new_type == ColorType::none) {
        ECVL_ERROR_WRONG_PARAMS("Cannot change color to or from ColorType::none")
    }

    if (src.colortype_ == new_type) {
        // if not, check if dst==src
        if (&src != &dst) { // if no, copy
            dst = src;
        }
        return;
    }

    src.hal_->ChangeColorSpace(src, dst, new_type);
}

void Threshold(const Image& src, Image& dst, double thresh, double maxval, ThresholdingType thresh_type)
{
    AlwaysCheck(src, dst);

    src.hal_->Threshold(src, dst, thresh, maxval, thresh_type);
}

std::vector<double> Histogram(const Image& src)
{
    if (src.IsEmpty()) {
        ECVL_ERROR_EMPTY_IMAGE
    }

    if (src.elemtype_ != DataType::uint8 || src.colortype_ != ColorType::GRAY) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }
    return src.hal_->Histogram(src);
}

int OtsuThreshold(const Image& src)
{
    if (src.IsEmpty()) {
        ECVL_ERROR_EMPTY_IMAGE
    }

    if (src.colortype_ != ColorType::GRAY) { // What if the Image has ColorType::none?
        ECVL_ERROR_WRONG_PARAMS("The OtsuThreshold requires a grayscale Image")
    }

    if (src.elemtype_ != DataType::uint8) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    return src.hal_->OtsuThreshold(src);
}

void Filter2D(const Image& src, Image& dst, const Image& ker, DataType type)
{
    AlwaysCheck(src, dst);

    if (src.channels_ != "xyc" || (ker.channels_ != "xyc" && ker.dims_[2] != 1) ||
        src.elemtype_ != DataType::uint8 || ker.elemtype_ != DataType::float64 || !ker.contiguous_ || !src.contiguous_) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if ((ker.dims_[0] % 2 == 0) || (ker.dims_[1] % 2 == 0)) {
        ECVL_ERROR_WRONG_PARAMS("Kernel sizes must be odd")
    }

    if (type == DataType::none) {
        type = src.elemtype_;
    }

    src.hal_->Filter2D(src, dst, ker, type);
}

void SeparableFilter2D(const Image& src, Image& dst, const vector<double>& kerX, const vector<double>& kerY, DataType type)
{
    AlwaysCheck(src, dst);

    if (src.channels_ != "xyc" || !src.contiguous_) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if ((kerX.size() % 2 == 0) || (kerY.size() % 2 == 0)) {
        ECVL_ERROR_WRONG_PARAMS("Kernel sizes must be odd")
    }

    if (type == DataType::none) {
        type = src.elemtype_;
    }

    src.hal_->SeparableFilter2D(src, dst, kerX, kerY, type);
}

void GaussianBlur(const Image& src, Image& dst, int sizeX, int sizeY, double sigmaX, double sigmaY)
{
    AlwaysCheck(src, dst);

    if (src.channels_ != "xyc") {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if (sizeX < 0 || (sizeX % 2 != 1)) {
        ECVL_ERROR_WRONG_PARAMS("sizeX must either be positive and odd or zero")
    }
    if (sizeY < 0 || (sizeY % 2 != 1)) {
        ECVL_ERROR_WRONG_PARAMS("sizeY must either be positive and odd or zero")
    }

    bool sigmaX_zero = false;
    if (sigmaX <= 0) {
        sigmaX_zero = true;
        if (sizeX == 0) {
            ECVL_ERROR_WRONG_PARAMS("sigmaX and sizeX can't be both 0")
        }
        else {
            sigmaX = 0.3 * ((sizeX - 1) * 0.5 - 1) + 0.8;
        }
    }
    if (sigmaY <= 0) {
        if (!sigmaX_zero) {
            sigmaY = sigmaX;
        }
        else if (sizeY == 0) {
            ECVL_ERROR_WRONG_PARAMS("sigmaX, sigmaY and sizeY can't be 0 at the same time")
        }
        else {
            sigmaY = 0.3 * ((sizeY - 1) * 0.5 - 1) + 0.8;
        }
    }

    src.hal_->GaussianBlur(src, dst, sizeX, sizeY, sigmaX, sigmaY);
}

void GaussianBlur(const Image& src, Image& dst, double sigma)
{
    // Formula from: https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
    int size = static_cast<int>(((sigma - 0.8) / 0.3 + 1) / 0.5 + 1);

    // Check if computed size is even
    size = size % 2 == 0 ? size + 1 : size;
    // Check if computed size is less than 3, the smallest kernel size allowed
    size = size < 3 ? 3 : size;

    GaussianBlur(src, dst, size, size, sigma, sigma);
}

void AdditiveLaplaceNoise(const Image& src, Image& dst, double std_dev)
{
    AlwaysCheck(src, dst);

    if (!src.contiguous_) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if (std_dev <= 0) {
        ECVL_ERROR_WRONG_PARAMS("std_dev must be >= 0")
    }

    src.hal_->AdditiveLaplaceNoise(src, dst, std_dev);
}

void AdditivePoissonNoise(const Image& src, Image& dst, double lambda)
{
    AlwaysCheck(src, dst);

    if (!src.contiguous_) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if (lambda <= 0) {
        ECVL_ERROR_WRONG_PARAMS("lambda must be >= 0")
    }

    src.hal_->AdditivePoissonNoise(src, dst, lambda);
}

void GammaContrast(const Image& src, Image& dst, double gamma)
{
    AlwaysCheck(src, dst);

    if (src.elemtype_ != DataType::uint8) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    src.hal_->GammaContrast(src, dst, gamma);
}

void CoarseDropout(const Image& src, Image& dst, double p, double drop_size, bool per_channel)
{
    AlwaysCheck(src, dst);

    if (src.channels_ != "xyc") {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    src.hal_->CoarseDropout(src, dst, p, drop_size, per_channel);
}

void IntegralImage(const Image& src, Image& dst, DataType dst_type)
{
    AlwaysCheck(src, dst);

    if (src.channels_ != "xyc" || src.Channels() != 1 || src.elemtype_ != DataType::uint8 || dst_type != DataType::float64) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    src.hal_->IntegralImage(src, dst, dst_type);
}

void NonMaximaSuppression(const Image& src, Image& dst)
{
    AlwaysCheck(src, dst);

    if (src.channels_ != "xyc" || src.Channels() != 1 || src.elemtype_ != DataType::int32) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    src.hal_->NonMaximaSuppression(src, dst);
}

vector<ecvl::Point2i> GetMaxN(const Image& src, size_t n)
{
    if (src.IsEmpty()) {
        ECVL_ERROR_EMPTY_IMAGE
    }

    if (src.channels_ != "xyc" || src.Channels() != 1 || src.elemtype_ != DataType::int32) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    return src.hal_->GetMaxN(src, n);
}

void ConnectedComponentsLabeling(const Image& src, Image& dst)
{
    AlwaysCheck(src, dst);

    if (src.dims_.size() != 3 || src.channels_ != "xyc" || src.Channels() != 1 || src.elemtype_ != DataType::uint8) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    src.hal_->ConnectedComponentsLabeling(src, dst);
}

void FindContours(const Image& src, vector<vector<ecvl::Point2i>>& contours)
{
    if (src.IsEmpty()) {
        ECVL_ERROR_EMPTY_IMAGE
    }

    if (src.dims_.size() != 3 || src.channels_ != "xyc" || src.Channels() != 1 || src.elemtype_ != DataType::uint8) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    src.hal_->FindContours(src, contours);
}

void Stack(const vector<Image>& src, Image& dst)
{
    const auto& src_0 = src[0];

    // Check if src images have the same dimensions
    for (int i = 0; i < vsize(src); ++i) {
        if (src[i].IsEmpty()) {
            ECVL_ERROR_EMPTY_IMAGE
        }
        if (src_0.Width() != src[i].Width() || src_0.Height() != src[i].Height()) {
            ECVL_ERROR_WRONG_PARAMS("Cannot stack images with different dimensions.")
        }
        if (src_0.dev_ != src[i].dev_) {
            ECVL_ERROR_DIFFERENT_DEVICES
        }
    }

    if (src_0.dev_ != dst.dev_ && dst.dev_ != Device::NONE) {
        ECVL_ERROR_DIFFERENT_DEVICES
    }

    src_0.hal_->Stack(src, dst);
}

void HConcat(const vector<Image>& src, Image& dst)
{
    const auto& src_0 = src[0];

    size_t c_pos = src_0.channels_.find('c');
    size_t x_pos = src_0.channels_.find('x');
    size_t y_pos = src_0.channels_.find('y');

    if (c_pos == string::npos || x_pos == string::npos || y_pos == string::npos) {
        ECVL_ERROR_WRONG_PARAMS("Malformed src image")
    }

    // Check if src images have the same y or c dimensions
    for (int i = 0; i < vsize(src); ++i) {
        if (src[i].IsEmpty()) {
            ECVL_ERROR_EMPTY_IMAGE
        }
        if (src_0.dims_[c_pos] != src[i].dims_[c_pos] || src_0.dims_[y_pos] != src[i].dims_[y_pos]) {
            ECVL_ERROR_WRONG_PARAMS("Cannot concatenate images with different dimensions.")
        }
        if (src_0.channels_ != src[i].channels_) {
            ECVL_ERROR_WRONG_PARAMS("Cannot concatenate images with different channels.")
        }
        if (src_0.dev_ != src[i].dev_) {
            ECVL_ERROR_DIFFERENT_DEVICES
        }
    }

    if (src_0.dev_ != dst.dev_ && dst.dev_ != Device::NONE) {
        ECVL_ERROR_DIFFERENT_DEVICES
    }

    src_0.hal_->HConcat(src, dst);
}

void VConcat(const vector<Image>& src, Image& dst)
{
    const auto& src_0 = src[0];

    size_t c_pos = src_0.channels_.find('c');
    size_t x_pos = src_0.channels_.find('x');
    size_t y_pos = src_0.channels_.find('y');

    if (c_pos == string::npos || x_pos == string::npos || y_pos == string::npos) {
        ECVL_ERROR_WRONG_PARAMS("Malformed src image")
    }

    // Check if src images have the same x or c dimensions
    for (int i = 0; i < vsize(src); ++i) {
        if (src[i].IsEmpty()) {
            ECVL_ERROR_EMPTY_IMAGE
        }
        if (src_0.dims_[c_pos] != src[i].dims_[c_pos] || src_0.dims_[x_pos] != src[i].dims_[x_pos]) {
            ECVL_ERROR_WRONG_PARAMS("Cannot concatenate images with different dimensions.")
        }
        if (src_0.channels_ != src[i].channels_) {
            ECVL_ERROR_WRONG_PARAMS("Cannot concatenate images with different channels.")
        }
        if (src_0.dev_ != src[i].dev_) {
            ECVL_ERROR_DIFFERENT_DEVICES
        }
    }

    if (src_0.dev_ != dst.dev_ && dst.dev_ != Device::NONE) {
        ECVL_ERROR_DIFFERENT_DEVICES
    }

    src_0.hal_->VConcat(src, dst);
}

void Morphology(const Image& src, Image& dst, MorphType op, Image& kernel, Point2i anchor, int iterations, BorderType border_type, const int& border_value)
{
    AlwaysCheck(src, dst);

    src.hal_->Morphology(src, dst, op, kernel, anchor, iterations, border_type, border_value);
}

void Inpaint(const Image& src, Image& dst, const Image& inpaintMask, double inpaintRadius, InpaintType flag)
{
    AlwaysCheck(src, dst);

    src.hal_->Inpaint(src, dst, inpaintMask, inpaintRadius, flag);
}

void MeanStdDev(const Image& src, std::vector<double>& mean, std::vector<double>& stddev)
{
    if (src.IsEmpty()) {
        ECVL_ERROR_EMPTY_IMAGE
    }

    src.hal_->MeanStdDev(src, mean, stddev);
}

void Transpose(const Image& src, Image& dst)
{
    AlwaysCheck(src, dst);

    src.hal_->Transpose(src, dst);
}

void GridDistortion(const Image& src, Image& dst, int num_steps, const std::array<float, 2>& distort_limit, InterpolationType interp, BorderType border_type, const int& border_value, const unsigned seed)
{
    AlwaysCheck(src, dst);

    src.hal_->GridDistortion(src, dst, num_steps, distort_limit, interp, border_type, border_value, seed);
}

void ElasticTransform(const Image& src, Image& dst, double alpha, double sigma, InterpolationType interp, BorderType border_type, const int& border_value, const unsigned seed)
{
    AlwaysCheck(src, dst);

    src.hal_->ElasticTransform(src, dst, alpha, sigma, interp, border_type, border_value, seed);
}

void OpticalDistortion(const Image& src, Image& dst, const std::array<float, 2>& distort_limit, const std::array<float, 2>& shift_limit, InterpolationType interp, BorderType border_type, const int& border_value, const unsigned seed)
{
    AlwaysCheck(src, dst);

    src.hal_->OpticalDistortion(src, dst, distort_limit, shift_limit, interp, border_type, border_value, seed);
}

void Salt(const Image& src, Image& dst, double p, bool per_channel, const unsigned seed)
{
    AlwaysCheck(src, dst);

    src.hal_->Salt(src, dst, p, per_channel, seed);
}

void Pepper(const Image& src, Image& dst, double p, bool per_channel, const unsigned seed)
{
    AlwaysCheck(src, dst);

    src.hal_->Pepper(src, dst, p, per_channel, seed);
}

void SaltAndPepper(const Image& src, Image& dst, double p, bool per_channel, const unsigned seed)
{
    AlwaysCheck(src, dst);

    src.hal_->SaltAndPepper(src, dst, p, per_channel, seed);
}

void SliceTimingCorrection(const Image& src, Image& dst, bool odd, bool down)
{
    AlwaysCheck(src, dst);

    if (src.channels_ != "xyzt") {
        ECVL_ERROR_WRONG_PARAMS("src must have 'xyzt' channels")
    }
    if (vsize(src.spacings_) != vsize(src.dims_)) {
        ECVL_ERROR_WRONG_PARAMS("src must have spacings")
    }

    src.hal_->SliceTimingCorrection(src, dst, odd, down);
}
} // namespace ecvl