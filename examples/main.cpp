#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../src/support_opencv.h"
#include "../src/imgcodecs.h"
#include "../src/filesystem.h"
#include "../src/imgproc.h"


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

    //Image rot1, rot2, rot3, rot4, rot5;
    //Rotate2D(img, rot5, 30, {});
    //Rotate2D(img, rot1, 30, {}, 0.5);
    //Rotate2D(img, rot2, 30, {}, 0.5, InterpolationType::cubic);
    //RotateFullImage2D(img, rot3, 30);
    //RotateFullImage2D(img, rot4, 30, 0.5, InterpolationType::cubic);

    //Resize(img, img, { 500,500 });
    //
    //Image out1, out2;
    //Flip2D(img, out1);
    //Mirror2D(img, out2);  

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

    Image img1, img2;
    ImRead("../data/Kodak/img0003.png", img1);
    ImRead("../data/Kodak/img0015.png", img2);

    ResizeScale(img1, img1, { 0.3, 0.3 });

    // A painfully slow idea for iterators
    for (auto it = img1.Begin<uint8_t>(); it != img1.End<uint8_t>(); ++it) {
        *it = static_cast<uint8_t>(*it * 0.5);
    }

    if (!ImWrite("test.jpg", img1)) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}