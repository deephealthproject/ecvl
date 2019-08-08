#include "ecvl/core/imgproc.h"

#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "ecvl/core/datatype_matrix.h"
#include "ecvl/core/standard_errors.h"

namespace ecvl {

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
        const uint8_t* g = src.data_ +                                         1    * src.strides_[c_pos];
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

double OtsuThreshold(const Image& src) {
    if (src.colortype_ != ColorType::GRAY) { // What if the Image has ColorType::none?
        throw std::runtime_error("The OtsuThreshold requires a grayscale Image");
    }

    if (true) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }
}

} // namespace ecvl