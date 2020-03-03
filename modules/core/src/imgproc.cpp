/*
* ECVL - European Computer Vision Library
* Version: 0.1
* copyright (c) 2020, Universit� degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include "ecvl/core/imgproc.h"
#include <chrono>
#ifdef ECVL_WITH_FPGA
#include "ecvl/core/imgproc_fpga.h"
#endif

#include <stdexcept>
#include <random>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "ecvl/core/datatype_matrix.h"
#include "ecvl/core/standard_errors.h"
#include <iostream>
namespace ecvl {
using namespace std;

/** @brief Given an InterpolationType, the GetOpenCVInterpolation function returns the associated OpenCV enum value.

@param[in] interp Interpolation type, see @ref InterpolationType.

@return Associated OpenCV enum value.
*/
static int GetOpenCVInterpolation(InterpolationType interp)
{
    switch (interp) {
    case InterpolationType::nearest:    return cv::INTER_NEAREST;
    case InterpolationType::linear:     return cv::INTER_LINEAR;
    case InterpolationType::area:       return cv::INTER_AREA;
    case InterpolationType::cubic:      return cv::INTER_CUBIC;
    case InterpolationType::lanczos4:   return cv::INTER_LANCZOS4;
    default:
        ECVL_ERROR_NOT_REACHABLE_CODE
    }
}

void ResizeDim(const ecvl::Image& src, ecvl::Image& dst, const std::vector<int>& newdims, InterpolationType interp)
{
    if (src.IsEmpty()) {
        ECVL_ERROR_EMPTY_IMAGE
    }

    if (src.channels_ == "xyc") {
        if (newdims.size() != 2) {
            throw std::runtime_error("Number of dimensions specified doesn't match image dimensions");
        }

#ifdef ECVL_WITH_FPGA
        cv::Mat src_mat = ImageToMat(src);
        cv::Mat m = cv::Mat::zeros(cv::Size(newdims[0], newdims[1]), CV_8UC(src_mat.channels()));
        ResizeDim_FPGA(src_mat, m, cv::Size(newdims[0], newdims[1]), GetOpenCVInterpolation(interp));
        dst = ecvl::MatToImage(m);

#else
        using namespace std::chrono;
        cv::Mat m;
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        cv::resize(ImageToMat(src), m, cv::Size(newdims[0], newdims[1]), 0.0, 0.0, GetOpenCVInterpolation(interp));
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        dst = ecvl::MatToImage(m);
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        std::cout << "Tiempo de ejecucion ResizeDim en cpu: " << time_span.count() << " seconds.";
        std::cout << std::endl;
#endif
    }
    else {
        ECVL_ERROR_NOT_IMPLEMENTED
    }
}

void ResizeScale(const Image& src, Image& dst, const std::vector<double>& scales, InterpolationType interp)
{
    if (src.IsEmpty()) {
        ECVL_ERROR_EMPTY_IMAGE
    }

    if (src.channels_ == "xyc") {
        if (scales.size() != 2) {
            throw std::runtime_error("Number of dimensions specified doesn't match image dimensions");
        }

        int nw = lround(src.dims_[0] * scales[0]);
        int nh = lround(src.dims_[1] * scales[1]);


#ifdef ECVL_WITH_FPGA
        printf("height %d\n", nh);
        cv::Mat src_mat = ImageToMat(src);
        cv::Mat m = cv::Mat::zeros(cv::Size(nw,nh), CV_8UC(src_mat.channels()));
        ResizeDim_FPGA(src_mat, m, cv::Size(nw,nh), GetOpenCVInterpolation(interp));
        dst = ecvl::MatToImage(m);
#else
        using namespace std::chrono;
        cv::Mat m;
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        cv::resize(ImageToMat(src), m, cv::Size(nw,nh), 0.0, 0.0, GetOpenCVInterpolation(interp));
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        dst = ecvl::MatToImage(m);
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        std::cout << "Tiempo de ejecucion ResizeDim en cpu: " << time_span.count() << " seconds.";
        std::cout << std::endl;
#endif
    }
    else {
        ECVL_ERROR_NOT_IMPLEMENTED
    }
}

void Flip2D(const ecvl::Image& src, ecvl::Image& dst)
{
    if (src.IsEmpty()) {
        ECVL_ERROR_EMPTY_IMAGE
    }

    if (src.channels_ == "xyc") {
        cv::Mat m;
        cv::flip(ImageToMat(src), m, 0);
        dst = ecvl::MatToImage(m);
    }
    else {
        ECVL_ERROR_NOT_IMPLEMENTED
    }
}

void Mirror2D(const ecvl::Image& src, ecvl::Image& dst)
{
    if (src.IsEmpty()) {
        ECVL_ERROR_EMPTY_IMAGE
    }

    if (src.channels_ == "xyc") {
        cv::Mat m;
        cv::flip(ImageToMat(src), m, 1);
        dst = ecvl::MatToImage(m);
    }
    else {
        ECVL_ERROR_NOT_IMPLEMENTED
    }
}

void Rotate2D(const ecvl::Image& src, ecvl::Image& dst, double angle, const std::vector<double>& center, double scale, InterpolationType interp)
{
    if (src.IsEmpty()) {
        ECVL_ERROR_EMPTY_IMAGE
    }

    cv::Point2f pt;
    if (center.empty()) {
        pt = { src.dims_[0] / 2.0f, src.dims_[1] / 2.0f };
    }
    else if (center.size() != 2) {
        throw std::runtime_error("Rotation center must have two dimensions");
    }
    else {
        pt = { float(center[0]), float(center[1]) };
    }

    if (src.channels_ == "xyc") {
        cv::Mat rot_matrix;
        rot_matrix = cv::getRotationMatrix2D(pt, -angle, scale);

#ifdef ECVL_WITH_FPGA
        cv::Mat src_mat = ImageToMat(src);
        cv::Mat m = cv::Mat::zeros({ src.dims_[0], src.dims_[1] }, CV_8UC(src_mat.channels()));
        warpTransform_FPGA(src_mat, m,rot_matrix,{ src.dims_[0], src.dims_[1] }, GetOpenCVInterpolation(interp));
        dst = ecvl::MatToImage(m);
#else
        cv::Mat m;
        cv::warpAffine(ImageToMat(src), m, rot_matrix, { src.dims_[0], src.dims_[1] }, GetOpenCVInterpolation(interp));
        dst = ecvl::MatToImage(m);
#endif

    }
    else {
        ECVL_ERROR_NOT_IMPLEMENTED
    }
}

void RotateFullImage2D(const ecvl::Image& src, ecvl::Image& dst, double angle, double scale, InterpolationType interp)
{
    if (src.IsEmpty()) {
        ECVL_ERROR_EMPTY_IMAGE
    }

    if (src.channels_ == "xyc") {
        cv::Point2f pt;
        pt = { src.dims_[0] / 2.0f, src.dims_[1] / 2.0f };

        cv::Mat1d rot_matrix;
        rot_matrix = cv::getRotationMatrix2D(pt, -angle, scale);

        int w = src.dims_[0];
        int h = src.dims_[1];

        double cos = abs(rot_matrix(0, 0));
        double sin = abs(rot_matrix(0, 1));

        int nw = lround((h * sin) + (w * cos));
        int nh = lround((h * cos) + (w * sin));

        rot_matrix(0, 2) += (nw / 2) - pt.x;
        rot_matrix(1, 2) += (nh / 2) - pt.y;

        cv::Mat m;
        cv::warpAffine(ImageToMat(src), m, rot_matrix, { nw, nh }, GetOpenCVInterpolation(interp));
        dst = ecvl::MatToImage(m);
    }
    else {
        ECVL_ERROR_NOT_IMPLEMENTED
    }
}

inline void RGB2GRAYGeneric(const uint8_t* r, const uint8_t* g, const uint8_t* b, uint8_t* dst, DataType dt)
{
    // 0.299 * R + 0.587 * G + 0.114 * B

#define DEREF(ptr, type)            *reinterpret_cast<TypeInfo_t<DataType::type>*>(ptr)
#define CONST_DEREF(ptr, type)      *reinterpret_cast<const TypeInfo_t<DataType::type>*>(ptr)

#define ECVL_TUPLE(type, ...) \
case DataType::type: DEREF(dst, type) = saturate_cast<DataType::type>(0.299 * CONST_DEREF(r, type) + 0.587 * CONST_DEREF(g, type) + 0.114 * CONST_DEREF(b, type));  break;

    switch (dt) {
#include "ecvl/core/datatype_existing_tuples.inc.h"
    }

#undef ECVL_TUPLE
#undef DEREF
#undef CONST_DEREF
}

void ChangeColorSpace(const Image& src, Image& dst, ColorType new_type)
{
    if (src.colortype_ == ColorType::none || new_type == ColorType::none) {
        throw std::runtime_error("Cannot change color to or from ColorType::none");
    }

    if (src.colortype_ == new_type) {
        // if not, check if dst==src
        if (&src != &dst) { // if no, copy
            dst = src;
        }
        return;
    }

    Image tmp;

    if (src.colortype_ == ColorType::HSV || new_type == ColorType::HSV
        ||
        src.colortype_ == ColorType::YCbCr || new_type == ColorType::YCbCr
        ||
        src.colortype_ == ColorType::RGBA || new_type == ColorType::RGBA
        ) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if (src.colortype_ == ColorType::GRAY) {
        if (new_type == ColorType::RGB || new_type == ColorType::BGR) {
            if (src.channels_ == "xyc") {
                auto dims = src.dims_;
                dims[2] = 3;
                tmp = Image(dims, src.elemtype_, "xyc", new_type);
                auto plane0 = tmp.data_ + 0 * tmp.strides_[2];
                auto plane1 = tmp.data_ + 1 * tmp.strides_[2];
                auto plane2 = tmp.data_ + 2 * tmp.strides_[2];
                if (src.contiguous_) {
                    memcpy(plane0, src.data_, src.datasize_);
                    memcpy(plane1, src.data_, src.datasize_);
                    memcpy(plane2, src.data_, src.datasize_);
                }
                else {
                    auto i = src.Begin<uint8_t>(), e = src.End<uint8_t>();
                    for (; i != e; ++i) {
                        memcpy(plane0, i.ptr_, src.elemsize_);
                        memcpy(plane1, i.ptr_, src.elemsize_);
                        memcpy(plane2, i.ptr_, src.elemsize_);
                        plane0 += src.elemsize_;
                        plane1 += src.elemsize_;
                        plane2 += src.elemsize_;
                    }
                }
            }
            else if (src.channels_ == "cxy") {
                auto dims = src.dims_;
                dims[0] = 3;
                tmp = Image(dims, src.elemtype_, "cxy", new_type);
                auto plane0 = tmp.data_ + 0 * tmp.strides_[0];
                auto plane1 = tmp.data_ + 1 * tmp.strides_[0];
                auto plane2 = tmp.data_ + 2 * tmp.strides_[0];
                auto i = src.Begin<uint8_t>(), e = src.End<uint8_t>();
                for (; i != e; ++i) {
                    memcpy(plane0, i.ptr_, src.elemsize_);
                    memcpy(plane1, i.ptr_, src.elemsize_);
                    memcpy(plane2, i.ptr_, src.elemsize_);
                    plane0 += 3 * src.elemsize_;
                    plane1 += 3 * src.elemsize_;
                    plane2 += 3 * src.elemsize_;
                }
            }
            else {
                ECVL_ERROR_NOT_IMPLEMENTED
            }
        }
        dst = std::move(tmp);
        return;
    }

    if ((src.colortype_ == ColorType::RGB || src.colortype_ == ColorType::BGR) && new_type == ColorType::GRAY) {
        size_t c_pos = src.channels_.find('c');
        if (c_pos == std::string::npos) {
            ECVL_ERROR_WRONG_PARAMS("Malformed src image")
        }

        std::vector<int> tmp_dims = src.dims_;
        tmp_dims[c_pos] = 1;

        tmp.Create(tmp_dims, src.elemtype_, src.channels_, ColorType::GRAY, src.spacings_);

        const uint8_t* r = src.data_ + ((src.colortype_ == ColorType::RGB) ? 0 : 2) * src.strides_[c_pos];
        const uint8_t* g = src.data_ + 1 * src.strides_[c_pos];
        const uint8_t* b = src.data_ + ((src.colortype_ == ColorType::RGB) ? 2 : 0) * src.strides_[c_pos];

        for (size_t tmp_pos = 0; tmp_pos < tmp.datasize_; tmp_pos += tmp.elemsize_) {
            int x = tmp_pos;
            int src_pos = 0;
            for (int i = tmp.dims_.size() - 1; i >= 0; i--) {
                if (i != c_pos) {
                    src_pos += (x / tmp.strides_[i]) * src.strides_[i];
                    x %= tmp.strides_[i];
                }
            }

            RGB2GRAYGeneric(r + src_pos, g + src_pos, b + src_pos, tmp.data_ + tmp_pos, src.elemtype_);
        }
        dst = tmp;
        return;
    }
    if (src.colortype_ == ColorType::BGR && new_type == ColorType::GRAY) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    //TODO: update with the operator+ for iterators
    if (src.colortype_ == ColorType::BGR && new_type == ColorType::RGB
        ||
        src.colortype_ == ColorType::RGB && new_type == ColorType::BGR) {
        if (src.channels_ == "xyc") {
            tmp = Image(src.dims_, src.elemtype_, "xyc", new_type);
            auto plane0 = tmp.data_ + 0 * tmp.strides_[2];
            auto plane1 = tmp.data_ + 1 * tmp.strides_[2];
            auto plane2 = tmp.data_ + 2 * tmp.strides_[2];
            if (src.contiguous_) {
                memcpy(plane0, src.data_ + 2 * src.strides_[2], src.strides_[2]);
                memcpy(plane1, src.data_ + 1 * src.strides_[2], src.strides_[2]);
                memcpy(plane2, src.data_ + 0 * src.strides_[2], src.strides_[2]);
            }
            else {
                ECVL_ERROR_NOT_IMPLEMENTED
            }
        }
        else if (src.channels_ == "cxy") {
            tmp = Image(src.dims_, src.elemtype_, "cxy", new_type);
            auto plane0 = tmp.data_ + 0 * tmp.strides_[0];
            auto plane1 = tmp.data_ + 1 * tmp.strides_[0];
            auto plane2 = tmp.data_ + 2 * tmp.strides_[0];
            auto i = src.Begin<uint8_t>(), e = src.End<uint8_t>();
            for (; i != e; ++i) {
                memcpy(plane0, i.ptr_ + 2 * src.elemsize_, src.elemsize_);
                memcpy(plane1, i.ptr_ + 1 * src.elemsize_, src.elemsize_);
                memcpy(plane2, i.ptr_ + 0 * src.elemsize_, src.elemsize_);
                plane0 += 3 * src.elemsize_;
                plane1 += 3 * src.elemsize_;
                plane2 += 3 * src.elemsize_;
                ++i; ++i;
            }
        }
        else {
            ECVL_ERROR_NOT_IMPLEMENTED
        }
        dst = std::move(tmp);
        return;
    }

    ECVL_ERROR_NOT_REACHABLE_CODE
}

void Threshold(const Image& src, Image& dst, double thresh, double maxval, ThresholdingType thresh_type)
{


    int t_type;
    switch (thresh_type) {
    case ecvl::ThresholdingType::BINARY:        t_type = cv::THRESH_BINARY;      break;
    case ecvl::ThresholdingType::BINARY_INV:    t_type = cv::THRESH_BINARY_INV;  break;
    default:
        ECVL_ERROR_NOT_REACHABLE_CODE
    }

#ifdef ECVL_WITH_FPGA
    cv::Mat src_mat1 = ImageToMat(src);
    //Only suport for threshold binary
    cv::Mat m = src_mat1;
    Threshold_FPGA(src_mat1, m, thresh, maxval);
    dst = ecvl::MatToImage(m);
#else
    using namespace std::chrono;
    cv::Mat m;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    cv::threshold(ImageToMat(src), m, thresh, maxval, t_type);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    dst = MatToImage(m);
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Tiempo de ejecucion Threshold en cpu: " << time_span.count() << " seconds.";
    std::cout << std::endl;
#endif
}

std::vector<double> Histogram(const Image& src)
{
    if (src.elemtype_ != DataType::uint8 || src.colortype_ != ColorType::GRAY) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    std::vector<double> hist(256, 0);

    ConstView<DataType::uint8> view(src);
    for (auto it = view.Begin(); it != view.End(); ++it) {
        hist[*it]++;
    }

    int total_pixels = std::accumulate(src.dims_.begin(), src.dims_.end(), 1, std::multiplies<int>());
    for (auto it = hist.begin(); it < hist.end(); it++) {
        *it /= total_pixels;
    }

    return hist;
}

int OtsuThreshold(const Image& src)
{
    if (src.colortype_ != ColorType::GRAY) { // What if the Image has ColorType::none?
        throw std::runtime_error("The OtsuThreshold requires a grayscale Image");
    }

    if (src.elemtype_ != DataType::uint8) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

#ifdef ECVL_WITH_FPGA
    int threshold = 0;
    threshold = OtsuThreshold_FPGA(ImageToMat(src));
#else
    std::vector<double> hist = Histogram(src);

    double mu_t = 0;
    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    for (size_t i = 1; i < hist.size(); i++) {
        mu_t += hist[i] * i;
    }

    double w_k = 0;
    double mu_k = 0;
    double sigma_max = 0;
    int threshold = 0;
    for (size_t k = 0; k < hist.size() - 1; k++) {
        w_k += hist[k];
        mu_k += hist[k] * k;

        double sigma = ((mu_t * w_k - mu_k) * (mu_t * w_k - mu_k)) / (w_k * (1 - w_k));
        if (sigma > sigma_max) {
            sigma_max = sigma;
            threshold = k;
        }
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Tiempo de ejecucion Otsu_Threshold en cpu: " << time_span.count() << " seconds.";
    std::cout << std::endl;
#endif

    return threshold;
}

void Filter2D(const Image& src, Image& dst, const Image& ker, DataType type)
{
    if (src.channels_ != "xyc" || (ker.channels_ != "xyc" && ker.dims_[2] != 1)) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if (src.elemtype_ != DataType::uint8) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if (ker.elemtype_ != DataType::float64) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if ((ker.dims_[0] % 2 == 0) || (ker.dims_[1] % 2 == 0)) {
        ECVL_ERROR_WRONG_PARAMS("Kernel sizes must be odd")
    }

    if (!ker.contiguous_ || !src.contiguous_) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if (type == DataType::none) {
        type = src.elemtype_;
    }

    Image tmp(src.dims_, type, src.channels_, src.colortype_, src.spacings_);

    int hlf_width = ker.dims_[0] / 2;
    int hlf_height = ker.dims_[1] / 2;

    TypeInfo_t<DataType::float64>* ker_data = reinterpret_cast<TypeInfo_t<DataType::float64>*>(ker.data_);

    uint8_t* tmp_ptr = tmp.data_;

    //auto tmp_it = tmp.ContiguousBegin<TypeInfo_t<DataType::uint8>>();
    //auto tmp_it_end = tmp.ContiguousEnd<TypeInfo_t<DataType::uint8>>();

    TypeInfo_t<DataType::uint8>* src_data = reinterpret_cast<TypeInfo_t<DataType::uint8>*>(src.data_);
    for (int chan = 0; chan < tmp.dims_[2]; chan++) {
        for (int r = 0; r < tmp.dims_[1]; r++) {
            for (int c = 0; c < tmp.dims_[0]; c++) {
                double acc = 0;
                int i = 0;
                for (int rk = 0; rk < ker.dims_[1]; rk++) {
                    for (int ck = 0; ck < ker.dims_[0]; ck++) {
                        int x = c + ck - hlf_width;
                        if (x < 0) x = 0; else if (x >= tmp.dims_[0]) x = tmp.dims_[0] - 1;

                        int y = r + rk - hlf_height;
                        if (y < 0) y = 0; else if (y >= tmp.dims_[1]) y = tmp.dims_[1] - 1;

                        acc += ker_data[i] * src_data[x + y * src.strides_[1]];

                        i++;
                    }
                }

#define ECVL_TUPLE(type, ...) \
case DataType::type: *reinterpret_cast<TypeInfo_t<DataType::type>*>(tmp_ptr) = static_cast<TypeInfo_t<DataType::type>>(acc); break;

                switch (type) {
#include "ecvl/core/datatype_existing_tuples.inc.h"
                }

#undef ECVL_TUPLE

                tmp_ptr += tmp.elemsize_;
            }
        }

        src_data += src.strides_[2] / sizeof(*src_data);
    }

    dst = tmp;
}

void SeparableFilter2D(const Image& src, Image& dst, const vector<double>& kerX, const vector<double>& kerY, DataType type)
{
    if (src.channels_ != "xyc") {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if (src.elemtype_ != DataType::uint8) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if ((kerX.size() % 2 == 0) || (kerY.size() % 2 == 0)) {
        ECVL_ERROR_WRONG_PARAMS("Kernel sizes must be odd")
    }

    if (!src.contiguous_) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if (type == DataType::none) {
        type = src.elemtype_;
    }

    Image tmp1(src.dims_, DataType::float64, src.channels_, src.colortype_, src.spacings_);
    Image tmp2(src.dims_, type, src.channels_, src.colortype_, src.spacings_);

    int hlf_width = kerX.size() / 2;
    int hlf_height = kerY.size() / 2;

    // X direction
    auto tmp1_it = tmp1.ContiguousBegin<TypeInfo_t<DataType::float64>>();
    TypeInfo_t<DataType::uint8>* src_data = reinterpret_cast<TypeInfo_t<DataType::uint8>*>(src.data_);
    for (int chan = 0; chan < tmp1.dims_[2]; chan++) {
        for (int r = 0; r < tmp1.dims_[1]; r++) {
            for (int c = 0; c < tmp1.dims_[0]; c++) {
                double acc = 0;
                for (unsigned int ck = 0; ck < kerX.size(); ck++) {
                    int x = c + ck - hlf_width;
                    if (x < 0) x = 0; else if (x >= tmp1.dims_[0]) x = tmp1.dims_[0] - 1;

                    acc += kerX[ck] * src_data[x];
                }

                *tmp1_it = acc;
                ++tmp1_it;
            }

            src_data += src.strides_[1] / sizeof(*src_data);
        }
    }

    uint8_t* tmp2_ptr = tmp2.data_;

    // Y direction
    TypeInfo_t<DataType::float64>* tmp1_data = reinterpret_cast<TypeInfo_t<DataType::float64>*>(tmp1.data_);
    for (int chan = 0; chan < tmp2.dims_[2]; chan++) {
        for (int r = 0; r < tmp2.dims_[1]; r++) {
            for (int c = 0; c < tmp2.dims_[0]; c++) {
                double acc = 0;
                for (unsigned int rk = 0; rk < kerY.size(); rk++) {
                    int y = r + rk - hlf_height;
                    if (y < 0) y = 0; else if (y >= tmp2.dims_[1]) y = tmp2.dims_[1] - 1;

                    acc += kerY[rk] * tmp1_data[c + y * tmp1.strides_[1] / sizeof(*tmp1_data)];
                }

#define ECVL_TUPLE(type, ...) \
case DataType::type: *reinterpret_cast<TypeInfo_t<DataType::type>*>(tmp2_ptr) = static_cast<TypeInfo_t<DataType::type>>(acc); break;

                switch (type) {
#include "ecvl/core/datatype_existing_tuples.inc.h"
                }

#undef ECVL_TUPLE

                tmp2_ptr += tmp2.elemsize_;
            }
        }

        tmp1_data += tmp1.strides_[2] / sizeof(*tmp1_data);
    }

    dst = tmp2;
}

void GaussianBlur(const Image& src, Image& dst, int sizeX, int sizeY, double sigmaX, double sigmaY)
{
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

    if (src.channels_ != "xyc") {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    // Find x kernel values
    vector<double> kernelX(sizeX);
    double sum = 0;
    for (int i = 0; i < sizeX; i++) {
        double coeff = exp(-((i - (sizeX - 1) / 2) * (i - (sizeX - 1) / 2)) / (2 * sigmaX * sigmaX));
        sum += coeff;
        kernelX[i] = coeff;
    }
    for (int i = 0; i < sizeX; i++) {
        kernelX[i] /= sum;
    }

    // Find y kernel values
    vector<double> kernelY(sizeY);
    sum = 0;
    for (int i = 0; i < sizeY; i++) {
        double coeff = exp(-((i - (sizeY - 1) / 2) * (i - (sizeY - 1) / 2)) / (2 * sigmaY * sigmaY));
        sum += coeff;
        kernelY[i] = coeff;
    }
    for (int i = 0; i < sizeY; i++) {
        kernelY[i] /= sum;
    }

    SeparableFilter2D(src, dst, kernelX, kernelY);
}

void AdditiveLaplaceNoise(const Image& src, Image& dst, double scale)
{
    if (!src.contiguous_) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if (scale <= 0) {
        ECVL_ERROR_WRONG_PARAMS("scale must be >= 0")
    }

    Image tmp(src.dims_, src.elemtype_, src.channels_, src.colortype_, src.spacings_);

    random_device rd;
    mt19937 gen(rd());
    exponential_distribution<> dist(1 / scale);

    for (uint8_t* tmp_ptr = tmp.data_, *src_ptr = src.data_; tmp_ptr < tmp.data_ + tmp.datasize_; tmp_ptr += tmp.elemsize_, src_ptr += src.elemsize_) {
        double exp1 = dist(gen);
        double exp2 = dist(gen);

        double noise = exp1 - exp2;

#define ECVL_TUPLE(type, ...) \
case DataType::type: *reinterpret_cast<TypeInfo_t<DataType::type>*>(tmp_ptr) = saturate_cast<TypeInfo_t<DataType::type>>(noise + *reinterpret_cast<TypeInfo_t<DataType::type>*>(src_ptr)); break;

        switch (tmp.elemtype_) {
#include "ecvl/core/datatype_existing_tuples.inc.h"
        }

#undef ECVL_TUPLE
    }

    dst = tmp;
}

void AdditivePoissonNoise(const Image& src, Image& dst, double lambda)
{
    if (!src.contiguous_) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if (lambda <= 0) {
        ECVL_ERROR_WRONG_PARAMS("lambda must be >= 0")
    }

    Image tmp(src.dims_, src.elemtype_, src.channels_, src.colortype_, src.spacings_);

    random_device rd;
    mt19937 gen(rd());
    poisson_distribution<> dist(lambda);

    for (uint8_t* tmp_ptr = tmp.data_, *src_ptr = src.data_; tmp_ptr < tmp.data_ + tmp.datasize_; tmp_ptr += tmp.elemsize_, src_ptr += src.elemsize_) {
        double noise = dist(gen);

#define ECVL_TUPLE(type, ...) \
case DataType::type: *reinterpret_cast<TypeInfo_t<DataType::type>*>(tmp_ptr) = saturate_cast<TypeInfo_t<DataType::type>>(noise + *reinterpret_cast<TypeInfo_t<DataType::type>*>(src_ptr)); break;

        switch (tmp.elemtype_) {
#include "ecvl/core/datatype_existing_tuples.inc.h"
        }

#undef ECVL_TUPLE
    }

    dst = tmp;
}

void GammaContrast(const Image& src, Image& dst, double gamma)
{
    if (src.elemtype_ != DataType::uint8) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    Image tmp(src.dims_, src.elemtype_, src.channels_, src.colortype_, src.spacings_);

    for (uint8_t* tmp_ptr = tmp.data_, *src_ptr = src.data_; tmp_ptr < tmp.data_ + tmp.datasize_; tmp_ptr += tmp.elemsize_, src_ptr += src.elemsize_) {
#define ECVL_TUPLE(type, ...) \
case DataType::type: *reinterpret_cast<TypeInfo_t<DataType::type>*>(tmp_ptr) = saturate_cast<TypeInfo_t<DataType::type>>(pow(*reinterpret_cast<TypeInfo_t<DataType::type>*>(src_ptr) / 255., gamma) * 255); break;

        switch (tmp.elemtype_) {
#include "ecvl/core/datatype_existing_tuples.inc.h"
        }

#undef ECVL_TUPLE
    }

    dst = tmp;
}

void CoarseDropout(const Image& src, Image& dst, double p, double drop_size, bool per_channel)
{
    if (src.channels_ != "xyc") {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    int rectX = static_cast<int>(src.dims_[0] * drop_size);
    int rectY = static_cast<int>(src.dims_[1] * drop_size);

    Image tmp = src;

    random_device rd;
    mt19937 gen(rd());
    discrete_distribution<> dist({ p, 1 - p });

    if (per_channel) {
        for (int ch = 0; ch < src.dims_[2]; ch++) {
            uint8_t* tmp_ptr = tmp.Ptr({ 0, 0, ch });

            for (int r = 0; r < src.dims_[1]; r += rectY) {
                for (int c = 0; c < src.dims_[0]; c += rectX) {
                    if (dist(gen) == 0) {
                        for (int rdrop = r; rdrop < r + rectY && rdrop < src.dims_[1]; rdrop++) {
                            for (int cdrop = c; cdrop < c + rectX && cdrop < src.dims_[0]; cdrop++) {
#define ECVL_TUPLE(type, ...) \
case DataType::type: *reinterpret_cast<TypeInfo_t<DataType::type>*>(tmp_ptr + rdrop * tmp.strides_[1] + cdrop * tmp.strides_[0]) = static_cast<TypeInfo_t<DataType::type>>(0); break;

                                switch (tmp.elemtype_) {
#include "ecvl/core/datatype_existing_tuples.inc.h"
                                }

#undef ECVL_TUPLE
                            }
                        }
                    }
                }
            }
        }
    }

    else {
        vector<uint8_t*> channel_ptrs;
        for (int ch = 0; ch < tmp.dims_[2]; ch++) {
            channel_ptrs.push_back(tmp.Ptr({ 0, 0, ch }));
        }

        for (int r = 0; r < src.dims_[1]; r += rectY) {
            for (int c = 0; c < src.dims_[0]; c += rectX) {
                if (dist(gen) == 0) {
                    for (int ch = 0; ch < src.dims_[2]; ch++) {
                        for (int rdrop = r; rdrop < r + rectY && rdrop < src.dims_[1]; rdrop++) {
                            for (int cdrop = c; cdrop < c + rectX && cdrop < src.dims_[0]; cdrop++) {
#define ECVL_TUPLE(type, ...) \
case DataType::type: *reinterpret_cast<TypeInfo_t<DataType::type>*>(channel_ptrs[ch] + rdrop * tmp.strides_[1] + cdrop * tmp.strides_[0]) = static_cast<TypeInfo_t<DataType::type>>(0); break;

                                switch (tmp.elemtype_) {
#include "ecvl/core/datatype_existing_tuples.inc.h"
                                }

#undef ECVL_TUPLE
                            }
                        }
                    }
                }
            }
        }
    }

    dst = tmp;
}

void IntegralImage(const Image& src, Image& dst, DataType dst_type)
{
    if (src.channels_ != "xyc" || src.Channels() != 1 || src.elemtype_ != DataType::uint8 || dst_type != DataType::float64) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    Image out({ src.dims_[0] + 1, src.dims_[1] + 1, src.dims_[2] }, dst_type, src.channels_, ColorType::GRAY);

    ConstContiguousViewXYC<DataType::uint8> vsrc(src);
    ContiguousViewXYC<DataType::float64> vdst(out);

    switch (out.elemtype_) {
    case DataType::float64:
        for (int y = 0; y < vdst.height(); ++y) {
            for (int x = 0; x < vdst.width(); ++x) {
                if (!x || !y) {
                    vdst(x, y, 0) = 0.;
                }
                else {
                    vdst(x, y, 0) = vsrc(x - 1, y - 1, 0) + vdst(x - 1, y, 0) + vdst(x, y - 1, 0) - vdst(x - 1, y - 1, 0);
                }
            }
        }
        break;
    }

    dst = std::move(out);
}

void NonMaximaSuppression(const Image& src, Image& dst)
{
    if (src.channels_ != "xyc" || src.Channels() != 1 || src.elemtype_ != DataType::int32) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    Image out(src);
    memset(out.data_, 0, out.datasize_);

    ConstContiguousViewXYC<DataType::int32> vsrc(src);
    ContiguousViewXYC<DataType::int32> vout(out);

    for (int y = 1; y < vout.height() - 1; ++y) {
        for (int x = 1; x < vout.width() - 1; ++x) {
            int cur = vsrc(x, y, 0);
            if (cur < vsrc(x - 1, y - 1, 0) ||
                cur < vsrc(x - 1, y, 0) ||
                cur < vsrc(x - 1, y + 1, 0) ||
                cur < vsrc(x, y - 1, 0) ||
                cur < vsrc(x, y + 1, 0) ||
                cur < vsrc(x + 1, y - 1, 0) ||
                cur < vsrc(x + 1, y, 0) ||
                cur < vsrc(x + 1, y + 1, 0)) {
                continue;
            }

            vout(x, y, 0) = cur;
        }
    }

    dst = std::move(out);
}

vector<ecvl::Point2i> GetMaxN(const Image& src, size_t n)
{
    if (src.channels_ != "xyc" || src.Channels() != 1 || src.elemtype_ != DataType::int32) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    ConstContiguousViewXYC<DataType::int32> vsrc(src);
    using pqt = pair<int32_t, Point2i>;
    vector<pqt> pq(n, make_pair(std::numeric_limits<int32_t>::min(), Point2i{ -1,-1 }));
    for (int y = 0; y < vsrc.height(); ++y) { // rows
        for (int x = 0; x < vsrc.width(); ++x) { // cols
            int32_t cur_value = vsrc(x, y, 0);
            if (cur_value > pq.front().first) {
                pop_heap(begin(pq), end(pq), greater<pqt>{});
                pq.back() = make_pair(cur_value, Point2i{ x,y });
                push_heap(begin(pq), end(pq), greater<pqt>{});
            }
        }
    }

    vector<ecvl::Point2i> max_coords;
    max_coords.reserve(n);
    for (const auto& x : pq) {
        max_coords.push_back(x.second);
    }
    return max_coords;
}

// Union-Find (UF) with path compression (PC) as in:
// Two Strategies to Speed up Connected Component Labeling Algorithms
// Kesheng Wu, Ekow Otoo, Kenji Suzuki
struct UFPC {
    // Maximum number of labels (included background) = 2^(sizeof(unsigned) x 8)
    unsigned* P_;
    unsigned length_;

    unsigned NewLabel()
    {
        P_[length_] = length_;
        return length_++;
    }

    unsigned Merge(unsigned i, unsigned j)
    {
        // FindRoot(i)
        unsigned root(i);
        while (P_[root] < root) {
            root = P_[root];
        }
        if (i != j) {
            // FindRoot(j)
            unsigned root_j(j);
            while (P_[root_j] < root_j) {
                root_j = P_[root_j];
            }
            if (root > root_j) {
                root = root_j;
            }
            // SetRoot(j, root);
            while (P_[j] < j) {
                unsigned t = P_[j];
                P_[j] = root;
                j = t;
            }
            P_[j] = root;
        }
        // SetRoot(i, root);
        while (P_[i] < i) {
            unsigned t = P_[i];
            P_[i] = root;
            i = t;
        }
        P_[i] = root;
        return root;
    }
};

void ConnectedComponentsLabeling(const Image& src, Image& dst)
{
    if (src.dims_.size() != 3 || src.channels_ != "xyc" || src.Channels() != 1 || src.elemtype_ != DataType::uint8) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    Image tmp(src.dims_, DataType::int32, "xyc", ColorType::GRAY);

    const int h = src.dims_[1];
    const int w = src.dims_[0];

    UFPC ufpc;
    unsigned*& P = ufpc.P_;
    unsigned& P_length = ufpc.length_;

    P = new unsigned[((size_t)((h + 1) / 2) * (size_t)((w + 1) / 2) + 1)];
    P[0] = 0;	 // First label is for background pixels
    P_length = 1;

    int e_rows = h & 0xfffffffe;
    bool o_rows = h % 2 == 1;
    int e_cols = w & 0xfffffffe;
    bool o_cols = w % 2 == 1;

    // We work with 2x2 blocks
    // +-+-+-+
    // |P|Q|R|
    // +-+-+-+
    // |S|X|
    // +-+-+

    // The pixels are named as follows
    // +---+---+---+
    // |a b|c d|e f|
    // |g h|i j|k l|
    // +---+---+---+
    // |m n|o p|
    // |q r|s t|
    // +---+---+

    // Pixels a, f, l, q are not needed, since we need to understand the
    // the connectivity between these blocks and those pixels only matter
    // when considering the outer connectivities

    // A bunch of defines used to check if the pixels are foreground,
    // without going outside the image limits.

    // First scan

// Define Conditions and Actions
    {
#define CONDITION_B img_row_prev_prev[c-1]>0
#define CONDITION_C img_row_prev_prev[c]>0
#define CONDITION_D img_row_prev_prev[c+1]>0
#define CONDITION_E img_row_prev_prev[c+2]>0

#define CONDITION_G img_row_prev[c-2]>0
#define CONDITION_H img_row_prev[c-1]>0
#define CONDITION_I img_row_prev[c]>0
#define CONDITION_J img_row_prev[c+1]>0
#define CONDITION_K img_row_prev[c+2]>0

#define CONDITION_M img_row[c-2]>0
#define CONDITION_N img_row[c-1]>0
#define CONDITION_O img_row[c]>0
#define CONDITION_P img_row[c+1]>0

#define CONDITION_R img_row_fol[c-1]>0
#define CONDITION_S img_row_fol[c]>0
#define CONDITION_T img_row_fol[c+1]>0

        // Action 1: No action
#define ACTION_1 img_labels_row[c] = 0;
                               // Action 2: New label (the block has foreground pixels and is not connected to anything else)
#define ACTION_2 img_labels_row[c] = ufpc.NewLabel();
                               //Action 3: Assign label of block P
#define ACTION_3 img_labels_row[c] = img_labels_row_prev_prev[c - 2];
                               // Action 4: Assign label of block Q
#define ACTION_4 img_labels_row[c] = img_labels_row_prev_prev[c];
                               // Action 5: Assign label of block R
#define ACTION_5 img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                               // Action 6: Assign label of block S
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 2];
                               // Action 7: Merge labels of block P and Q
#define ACTION_7 img_labels_row[c] = ufpc.Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]);
                               //Action 8: Merge labels of block P and R
#define ACTION_8 img_labels_row[c] = ufpc.Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]);
                               // Action 9 Merge labels of block P and S
#define ACTION_9 img_labels_row[c] = ufpc.Merge(img_labels_row_prev_prev[c - 2], img_labels_row[c - 2]);
                               // Action 10 Merge labels of block Q and R
#define ACTION_10 img_labels_row[c] = ufpc.Merge(img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]);
                               // Action 11: Merge labels of block Q and S
#define ACTION_11 img_labels_row[c] = ufpc.Merge(img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                               // Action 12: Merge labels of block R and S
#define ACTION_12 img_labels_row[c] = ufpc.Merge(img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                               // Action 13: Merge labels of block P, Q and R
#define ACTION_13 img_labels_row[c] = ufpc.Merge(ufpc.Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]), img_labels_row_prev_prev[c + 2]);
                               // Action 14: Merge labels of block P, Q and S
#define ACTION_14 img_labels_row[c] = ufpc.Merge(ufpc.Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]), img_labels_row[c - 2]);
                               //Action 15: Merge labels of block P, R and S
#define ACTION_15 img_labels_row[c] = ufpc.Merge(ufpc.Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                               //Action 16: labels of block Q, R and S
#define ACTION_16 img_labels_row[c] = ufpc.Merge(ufpc.Merge(img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
    }

    if (h == 1) {
        // Single line
        int r = 0;
        const unsigned char* const img_row = src.Ptr({ 0, 0, 0 });
        unsigned* const img_labels_row = reinterpret_cast<unsigned*>(tmp.Ptr({ 0, 0, 0 }));
        int c = -2;
#include "labeling_bolelli_2019_forest_singleline.inc"
    }
    else {
        // More than one line

        // First couple of lines
        {
            int r = 0;
            const unsigned char* const img_row = src.Ptr({ 0, 0, 0 });
            unsigned* const img_labels_row = reinterpret_cast<unsigned*>(tmp.Ptr({ 0, 0, 0 }));
            const unsigned char* const img_row_fol = img_row + src.strides_[1];

            int c = -2;

#include "labeling_bolelli_2019_forest_firstline.inc"
        }

        // Every other line but the last one if image has an odd number of rows
        for (int r = 2; r < e_rows; r += 2) {
            // Get rows pointer
            const unsigned char* const img_row = src.Ptr({ 0, r, 0 });
            const unsigned char* const img_row_prev = img_row - src.strides_[1];
            const unsigned char* const img_row_prev_prev = img_row_prev - src.strides_[1];
            const unsigned char* const img_row_fol = img_row + src.strides_[1];
            unsigned* const img_labels_row = reinterpret_cast<unsigned*>(tmp.Ptr({ 0, r, 0 }));
            unsigned* const img_labels_row_prev_prev = reinterpret_cast<unsigned*>((reinterpret_cast<uint8_t*>(img_labels_row) - 2 * tmp.strides_[1]));

            int c = -2;
            goto tree_0;

#include "labeling_bolelli_2019_forest.inc"
        }

        // Last line (in case the rows are odd)
        if (o_rows) {
            int r = h - 1;
            const unsigned char* const img_row = src.Ptr({ 0, r, 0 });
            const unsigned char* const img_row_prev = img_row - src.strides_[1];
            const unsigned char* const img_row_prev_prev = img_row_prev - src.strides_[1];
            unsigned* const img_labels_row = reinterpret_cast<unsigned*>(tmp.Ptr({ 0, r, 0 }));
            unsigned* const img_labels_row_prev_prev = reinterpret_cast<unsigned*>((reinterpret_cast<uint8_t*>(img_labels_row) - 2 * tmp.strides_[1]));

            int c = -2;
#include "labeling_bolelli_2019_forest_lastline.inc"
        }
    }

    // Undef Conditions and Actions
    {
#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8
#undef ACTION_9
#undef ACTION_10
#undef ACTION_11
#undef ACTION_12
#undef ACTION_13
#undef ACTION_14
#undef ACTION_15
#undef ACTION_16

#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E

#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K

#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P

#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
    }

    // Flatten
    unsigned k = 1;
    for (unsigned i = 1; i < P_length; ++i) {
        if (P[i] < i) {
            P[i] = P[P[i]];
        }
        else {
            P[i] = k;
            k = k + 1;
        }
    }

    unsigned int n_labels_ = k;

    // Second scan
    int r = 0;
    for (; r < e_rows; r += 2) {
        // Get rows pointer
        const unsigned char* const img_row = src.Ptr({ 0, r, 0 });
        const unsigned char* const img_row_fol = img_row + src.strides_[1];

        unsigned* const img_labels_row = reinterpret_cast<unsigned*>(tmp.Ptr({ 0, r, 0 }));
        unsigned* const img_labels_row_fol = reinterpret_cast<unsigned*>((reinterpret_cast<uint8_t*>(img_labels_row) + tmp.strides_[1]));

        int c = 0;
        for (; c < e_cols; c += 2) {
            int iLabel = img_labels_row[c];
            if (iLabel > 0) {
                iLabel = P[iLabel];
                if (img_row[c] > 0)
                    img_labels_row[c] = iLabel;
                else
                    img_labels_row[c] = 0;
                if (img_row[c + 1] > 0)
                    img_labels_row[c + 1] = iLabel;
                else
                    img_labels_row[c + 1] = 0;
                if (img_row_fol[c] > 0)
                    img_labels_row_fol[c] = iLabel;
                else
                    img_labels_row_fol[c] = 0;
                if (img_row_fol[c + 1] > 0)
                    img_labels_row_fol[c + 1] = iLabel;
                else
                    img_labels_row_fol[c + 1] = 0;
            }
            else {
                img_labels_row[c] = 0;
                img_labels_row[c + 1] = 0;
                img_labels_row_fol[c] = 0;
                img_labels_row_fol[c + 1] = 0;
            }
        }
        // Last column if the number of columns is odd
        if (o_cols) {
            int iLabel = img_labels_row[c];
            if (iLabel > 0) {
                iLabel = P[iLabel];
                if (img_row[c] > 0)
                    img_labels_row[c] = iLabel;
                else
                    img_labels_row[c] = 0;
                if (img_row_fol[c] > 0)
                    img_labels_row_fol[c] = iLabel;
                else
                    img_labels_row_fol[c] = 0;
            }
            else {
                img_labels_row[c] = 0;
                img_labels_row_fol[c] = 0;
            }
        }
    }
    // Last row if the number of rows is odd
    if (o_rows) {
        // Get rows pointer
        const unsigned char* const img_row = src.Ptr({ 0, r, 0 });
        unsigned* const img_labels_row = reinterpret_cast<unsigned*>(tmp.Ptr({ 0, r, 0 }));

        int c = 0;
        for (; c < e_cols; c += 2) {
            int iLabel = img_labels_row[c];
            if (iLabel > 0) {
                iLabel = P[iLabel];
                if (img_row[c] > 0)
                    img_labels_row[c] = iLabel;
                else
                    img_labels_row[c] = 0;
                if (img_row[c + 1] > 0)
                    img_labels_row[c + 1] = iLabel;
                else
                    img_labels_row[c + 1] = 0;
            }
            else {
                img_labels_row[c] = 0;
                img_labels_row[c + 1] = 0;
            }
        }
        // Last column if the number of columns is odd
        if (o_cols) {
            int iLabel = img_labels_row[c];
            if (iLabel > 0) {
                iLabel = P[iLabel];
                if (img_row[c] > 0)
                    img_labels_row[c] = iLabel;
                else
                    img_labels_row[c] = 0;
            }
            else {
                img_labels_row[c] = 0;
            }
        }
    }

    delete[] P;

    dst = move(tmp);
}

void FindContours(const Image& src, vector<vector<ecvl::Point2i>>& contours)
{
    if (src.dims_.size() != 3 || src.channels_ != "xyc" || src.Channels() != 1 || src.elemtype_ != DataType::uint8) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    cv::Mat cv_src = ecvl::ImageToMat(src);

    vector<vector<cv::Point>> cv_contours;
    vector<cv::Vec4i> hierarchy;
#if OpenCV_VERSION_MAJOR > 3
    cv::findContours(cv_src, cv_contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
#else
    cv::findContours(cv_src, cv_contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
#endif // OpenCV_VERSION_MAJOR > 3

    contours.resize(cv_contours.size());
    for (int i = 0; i < cv_contours.size(); ++i) {
        vector<ecvl::Point2i> t(cv_contours[i].size());
        for (int j = 0; j < cv_contours[i].size(); ++j) {
            t[j] = Point2i{ cv_contours[i][j].x, cv_contours[i][j].y };
        }
        contours[i] = t;
    }
}
} // namespace ecvl
