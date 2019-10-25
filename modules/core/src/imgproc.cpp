#include "ecvl/core/imgproc.h"

#include <stdexcept>
#include <random>

#include <opencv2/imgproc.hpp>

#include "ecvl/core/datatype_matrix.h"
#include "ecvl/core/standard_errors.h"

namespace ecvl {

using namespace std;

/** @brief Given an InterpolationType, the GetOpenCVInterpolation function returns the associated OpenCV enum value.

@param[in] interp Interpolation type, see @ref InterpolationType.

@return Associated OpenCV enum value.
*/
static int GetOpenCVInterpolation(InterpolationType interp) {
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

        cv::Mat m;
        cv::resize(ImageToMat(src), m, cv::Size(newdims[0], newdims[1]), 0.0, 0.0, GetOpenCVInterpolation(interp));
        dst = ecvl::MatToImage(m);
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

        cv::Mat m;
        cv::resize(ImageToMat(src), m, cv::Size(nw, nh), 0.0, 0.0, GetOpenCVInterpolation(interp));
        dst = ecvl::MatToImage(m);
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
        cv::Mat m;
        cv::warpAffine(ImageToMat(src), m, rot_matrix, { src.dims_[0], src.dims_[1] }, GetOpenCVInterpolation(interp));
        dst = ecvl::MatToImage(m);
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


inline void RGB2GRAYGeneric(const uint8_t* r, const uint8_t* g, const uint8_t* b, uint8_t* dst, DataType dt) {

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

void Threshold(const Image& src, Image& dst, double thresh, double maxval, ThresholdingType thresh_type) {
    cv::Mat m;

    int t_type;
    switch (thresh_type)
    {
    case ecvl::ThresholdingType::BINARY:        t_type = cv::THRESH_BINARY;      break;
    case ecvl::ThresholdingType::BINARY_INV:    t_type = cv::THRESH_BINARY_INV;  break;
    default:
        ECVL_ERROR_NOT_REACHABLE_CODE
    }

    cv::threshold(ImageToMat(src), m, thresh, maxval, t_type);
    dst = MatToImage(m);
}

std::vector<double> Histogram(const Image& src) {

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

int OtsuThreshold(const Image& src) {
    if (src.colortype_ != ColorType::GRAY) { // What if the Image has ColorType::none?
        throw std::runtime_error("The OtsuThreshold requires a grayscale Image");
    }

    if (src.elemtype_ != DataType::uint8) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    std::vector<double> hist = Histogram(src);

    double mu_t = 0;
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

    return threshold;
}


void Filter2D(const Image& src, Image& dst, const Image& ker, DataType type) {

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

void SeparableFilter2D(const Image& src, Image& dst, const vector<double>& kerX, const vector<double>& kerY, DataType type) {

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


void GaussianBlur(const Image& src, Image& dst, int sizeX, int sizeY, double sigmaX, double sigmaY) {

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

void AdditiveLaplaceNoise(const Image& src, Image& dst, double scale) {

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

void GammaContrast(const Image& src, Image& dst, double gamma) {

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

void CoarseDropout(const Image& src, Image& dst, double p, double drop_size, bool per_channel) {

    if (src.channels_ != "xyc") {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    int rectX = src.dims_[0] * drop_size;
    int rectY = src.dims_[1] * drop_size;

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

} // namespace ecvl