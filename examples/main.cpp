#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../src/support_opencv.h"
#include "../src/imgcodecs.h"
#include "../src/filesystem.h"

namespace ecvl {

enum class InterpolationType {
    nearest,
    linear,
    area,
    cubic,
    lanczos4
};
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
void Resize(const ecvl::Image& src, ecvl::Image& dst, const std::vector<int>& newdims, InterpolationType interp = InterpolationType::linear)
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

void Rotate2D(const ecvl::Image& src, ecvl::Image& dst, double angle, const std::vector<double>& center = {}, double scale = 1.0, InterpolationType interp = InterpolationType::linear)
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

void RotateFullImage2D(const ecvl::Image& src, ecvl::Image& dst, double angle, double scale = 1.0, InterpolationType interp = InterpolationType::linear)
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

int main(void)
{
    using namespace ecvl;
    using namespace filesystem;

    /*
    Image test({ 5, 5 }, DataType::uint16, "xy", ColorType::none);
    Img<uint16_t> t(test);
    t(0, 0) = 1;
    t(1, 0) = 2;
    t(2, 0) = 3;

    cv::Mat3b m(3, 2);
    m << cv::Vec3b(1, 1, 1), cv::Vec3b(2, 2, 2),
        cv::Vec3b(3, 3, 3), cv::Vec3b(4, 4, 4),
        cv::Vec3b(5, 5, 5), cv::Vec3b(6, 6, 6);
    Image img = MatToImage(m);
    */

    // Read ECVL image from file
    Image img;
    if (!ImRead("../data/test.jpg", img)) {
        return EXIT_FAILURE;
    }

    //Resize(img, img, { 500,500 });

    /*
    Image out1, out2;
    Flip2D(img, out1);
    Mirror2D(img, out2);
    */

    Image cropped;
    std::vector<int> start{ 100, 100, 2 };
    std::vector<int> size{ 200, 200, -1 };
    cropped.elemtype_ = img.elemtype_;
    cropped.elemsize_ = img.elemsize_;
    int ssize = size.size();
    for (int i = 0; i < ssize; ++i) {
        if (start[i] < 0 || start[i] >= img.dims_[i])
            throw std::runtime_error("Start of crop outside image limits");
        cropped.dims_.push_back(img.dims_[i] - start[i]);
        if (size[i] > cropped.dims_[i]) {
            throw std::runtime_error("Crop outside image limits");
        }
        if (size[i] >= 0) {
            cropped.dims_[i] = size[i];
        }
    }
    cropped.strides_ = img.strides_;
    cropped.channels_ = 1;
    cropped.colortype_ = ColorType::GRAY;
    cropped.data_ = img.Ptr(start);
    cropped.datasize_ = 0;
    cropped.contiguous_ = false;
    cropped.meta_ = img.meta_;
    cropped.mem_ = ShallowMemoryManager::GetInstance();



    /*
    Image rot1, rot2, rot3, rot4;
    Rotate2D(img, rot1, 30, {}, 0.5);
    Rotate2D(img, rot2, 30, {}, 0.5, InterpolationType::cubic);
    RotateFullImage2D(img, rot3, 30, 0.5);
    RotateFullImage2D(img, rot4, 30, 0.5, InterpolationType::cubic);
    */

    if (!ImWrite("test.jpg", img)) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}