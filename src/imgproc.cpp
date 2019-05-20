#include "imgproc.h"

#include <stdexcept>

//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace ecvl {

/* @brief Given an InterpolationType, the GetOpenCVInterpolation function returns the associated OpenCV enum value.

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
        throw std::runtime_error("Should not happen");
    }
}

void ResizeDim(const ecvl::Image& src, ecvl::Image& dst, const std::vector<int>& newdims, InterpolationType interp)
{
    if (src.IsEmpty()) {
        throw std::runtime_error("Empty image provided");
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
        throw std::runtime_error("Not implemented");
    }
}

void ResizeScale(const Image& src, Image& dst, const std::vector<double>& scales, InterpolationType interp)
{
    if (src.IsEmpty()) {
        throw std::runtime_error("Empty image provided");
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
        throw std::runtime_error("Not implemented");
    }
}

void Flip2D(const ecvl::Image& src, ecvl::Image& dst)
{
    if (src.IsEmpty()) {
        throw std::runtime_error("Empty image provided");
    }

    if (src.channels_ == "xyc") {
        cv::Mat m;
        cv::flip(ImageToMat(src), m, 0);
        dst = ecvl::MatToImage(m);
    }
    else {
        throw std::runtime_error("Not implemented");
    }
}

void Mirror2D(const ecvl::Image& src, ecvl::Image& dst)
{
    if (src.IsEmpty()) {
        throw std::runtime_error("Empty image provided");
    }

    if (src.channels_ == "xyc") {
        cv::Mat m;
        cv::flip(ImageToMat(src), m, 1);
        dst = ecvl::MatToImage(m);
    }
    else {
        throw std::runtime_error("Not implemented");
    }
}

void Rotate2D(const ecvl::Image& src, ecvl::Image& dst, double angle, const std::vector<double>& center, double scale, InterpolationType interp)
{
    if (src.IsEmpty()) {
        throw std::runtime_error("Empty image provided");
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
        throw std::runtime_error("Not implemented");
    }
}

void RotateFullImage2D(const ecvl::Image& src, ecvl::Image& dst, double angle, double scale, InterpolationType interp)
{
    if (src.IsEmpty()) {
        throw std::runtime_error("Empty image provided");
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
        throw std::runtime_error("Not implemented");
    }
}

} // namespace ecvl