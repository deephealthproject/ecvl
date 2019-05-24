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

namespace ecvl {

template<typename ViewType>
ecvl::Image& Mul(ecvl::Image& img, double d)
{
    using namespace ecvl;
    ViewType v(img);
    auto i = v.Begin(), e = v.End();
    for (; i != e; ++i) {
        auto &p = *i;
        p = static_cast<typename ViewType::basetype>(p * d);
    }
    return img;
}

ecvl::Image& Mul(ecvl::Image& img, double d)
{
    using namespace ecvl;
    if (img.contiguous_) {
        switch (img.elemtype_)
        {
#define ECVL_TUPLE(name, ...) case DataType::name: return Mul<ContiguousView<DataType::name>>(img, d);
#include "../src/datatype_existing_tuples.inc"
#undef ECVL_TUPLE
        default:
            throw std::runtime_error("Not implemented");
        }
    }
    else {
        switch (img.elemtype_)
        {
#define ECVL_TUPLE(name, ...) case DataType::name: return Mul<View<DataType::name>>(img, d);
#include "../src/datatype_existing_tuples.inc"
#undef ECVL_TUPLE
        default:
            throw std::runtime_error("Not implemented");
        }
    }
}

} // namespace ecvl

namespace ecvl{

void RearrangeChannels(const Image& in, Image& out, const std::string& channels)
{
    // Check if rearranging is required
    if (in.channels_ == channels) {
        // if not, check if out==in
        if (&in != &out) { // if no, copy            
            out = in;
        }
        return;
    }
    Image tmp;
    // Check if rearranging is possible, else throw
    if (in.channels_ == "xyc" && channels == "cxy") {
        tmp = Image({ in.dims_[2], in.dims_[0], in.dims_[1] }, in.elemtype_, channels, in.colortype_);
        auto i = in.Begin<uint8_t>();
        auto plane_elems = in.dims_[0] * in.dims_[1];
        for (int ch = 0; ch < in.dims_[2]; ++ch) {
            auto ptr = tmp.data_ + ch;
            for (int el = 0; el < plane_elems; ++el) {
                memcpy(ptr, i.ptr_, tmp.elemsize_);
                ++i;
                ptr += tmp.strides_[1];
            }
        }
    }
    else if(in.channels_ == "cxy" && channels == "xyc")
    {
        tmp = Image({ in.dims_[1], in.dims_[2], in.dims_[0] }, in.elemtype_, channels, in.colortype_);
        auto i = in.Begin<uint8_t>();
        auto plane_elems = in.dims_[0] * in.dims_[1];
        for (int el = 0; el < plane_elems; ++el) {
            auto ptr = tmp.data_ + el;
            for (int ch = 0; ch < in.dims_[2]; ++ch) {
                memcpy(ptr, i.ptr_, tmp.elemsize_);
                ++i;
                ptr += tmp.strides_[2];
            }
        }
    }
    else {
        throw std::runtime_error("Not implemented");
    }
    
    out = std::move(tmp);
}

} // namespace ecvl

int main(void)
{
    std::vector<int> test{ 1,2,3 };

    using namespace ecvl;
    using namespace filesystem;
    
    Image x({ 5, 5, 3 }, DataType::uint8, "xyc", ColorType::RGB);
    View<DataType::uint8> y(x);
    y({ 0,0,0 }) = 15; y({ 1,0,0 }) = 16; y({ 2,0,0 }) = 17; y({ 3,0,0 }) = 18; y({ 4,0,0 }) = 19;
    y({ 0,1,0 }) = 25; y({ 1,1,0 }) = 26; y({ 2,1,0 }) = 27; y({ 3,1,0 }) = 28; y({ 4,1,0 }) = 29;
    y({ 0,2,0 }) = 35; y({ 1,2,0 }) = 36; y({ 2,2,0 }) = 37; y({ 3,2,0 }) = 38; y({ 4,2,0 }) = 39;
    y({ 0,3,0 }) = 45; y({ 1,3,0 }) = 46; y({ 2,3,0 }) = 47; y({ 3,3,0 }) = 48; y({ 4,3,0 }) = 49;
    y({ 0,4,0 }) = 55; y({ 1,4,0 }) = 56; y({ 2,4,0 }) = 57; y({ 3,4,0 }) = 58; y({ 4,4,0 }) = 59;

    y({ 0,0,1 }) = 17; y({ 1,0,1 }) = 16; y({ 2,0,1 }) = 10; y({ 3,0,1 }) = 17; y({ 4,0,1 }) = 19;
    y({ 0,1,1 }) = 27; y({ 1,1,1 }) = 26; y({ 2,1,1 }) = 20; y({ 3,1,1 }) = 27; y({ 4,1,1 }) = 29;
    y({ 0,2,1 }) = 37; y({ 1,2,1 }) = 36; y({ 2,2,1 }) = 30; y({ 3,2,1 }) = 37; y({ 4,2,1 }) = 39;
    y({ 0,3,1 }) = 47; y({ 1,3,1 }) = 46; y({ 2,3,1 }) = 40; y({ 3,3,1 }) = 47; y({ 4,3,1 }) = 49;
    y({ 0,4,1 }) = 57; y({ 1,4,1 }) = 56; y({ 2,4,1 }) = 50; y({ 3,4,1 }) = 57; y({ 4,4,1 }) = 59;

    y({ 0,0,2 }) = 15; y({ 1,0,2 }) = 17; y({ 2,0,2 }) = 17; y({ 3,0,2 }) = 18; y({ 4,0,2 }) = 17;
    y({ 0,1,2 }) = 25; y({ 1,1,2 }) = 27; y({ 2,1,2 }) = 27; y({ 3,1,2 }) = 28; y({ 4,1,2 }) = 27;
    y({ 0,2,2 }) = 35; y({ 1,2,2 }) = 37; y({ 2,2,2 }) = 37; y({ 3,2,2 }) = 38; y({ 4,2,2 }) = 37;
    y({ 0,3,2 }) = 45; y({ 1,3,2 }) = 47; y({ 2,3,2 }) = 47; y({ 3,3,2 }) = 48; y({ 4,3,2 }) = 47;
    y({ 0,4,2 }) = 55; y({ 1,4,2 }) = 57; y({ 2,4,2 }) = 57; y({ 3,4,2 }) = 58; y({ 4,4,2 }) = 57;

    Image k;
    RearrangeChannels(x, k, "cxy");

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
    cropped.channels_ = img.channels_;
    cropped.colortype_ = ColorType::GRAY;
    cropped.data_ = img.Ptr(start);
    cropped.datasize_ = 0;
    cropped.contiguous_ = false;
    cropped.meta_ = img.meta_;
    cropped.mem_ = ShallowMemoryManager::GetInstance();

    ImWrite("cropped.png", cropped);

    Image img1, img2;
    ImRead("../data/Kodak/img0003.png", img1);
    ImRead("../data/Kodak/img0015.png", img2);

    //ResizeScale(img1, img1, { 0.3, 0.3 });

    Mul(img1, 0.5);

    cv::TickMeter tm;

    tm.reset();
    tm.start();
    Image add(img1.dims_, DataType::float64, "xyc", ColorType::none);
    ContiguousView<DataType::float64> v(add);
    ContiguousView<DataType::uint8> v1(img1);
    ContiguousView<DataType::uint8> v2(img2);
    {
        auto i = v.Begin(), e = v.End();
        auto i1 = v1.Begin(), e1 = v1.End();
        auto i2 = v2.Begin(), e2 = v2.End();
        for (; i1 != e1 && i2 != e2; ++i, ++i1, ++i2) {
            auto &p = *i;
            auto &p1 = *i1;
            auto &p2 = *i2;
            p = p1 / 2.0 + p2 / 2.0;
        }
    }
    tm.stop();
    std::cout << "Elapsed " << tm.getTimeSec() << " s\n";


    tm.reset();
    tm.start();
    for (auto i = img1.Begin<uint8_t>(), e = img1.End<uint8_t>(); i != e; ++i) {
        auto& p = *i;
        p = static_cast<uint8_t>(p * 0.5);
    }
    tm.stop();
    std::cout << "Elapsed " << tm.getTimeSec() << " s\n";

    tm.reset();
    tm.start();
    for (auto i = img1.ContiguousBegin<uint8_t>(), e = img1.ContiguousEnd<uint8_t>(); i != e; ++i) {
        auto& p = *i;
        p = static_cast<uint8_t>(p * 0.5);
    }
    tm.stop();
    std::cout << "Elapsed " << tm.getTimeSec() << " s\n";

    ContiguousView<DataType::uint8> view1(img1);
    tm.reset();
    tm.start();
    for (auto i = view1.Begin(), e = view1.End(); i != e; ++i) {
        auto& p = *i;
        p = static_cast<uint8_t>(p * 0.5);
    }
    tm.stop();
    std::cout << "Elapsed " << tm.getTimeSec() << " s\n";

    /*if (!ImWrite("test.jpg", img1)) {
        return EXIT_FAILURE;
    }*/
    return EXIT_SUCCESS;
}