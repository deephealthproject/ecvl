#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ecvl/core.h"
#include "ecvl/core/arithmetic.h"
//#include "ecvl/gui.h"

int main(void)
{
    using namespace ecvl;
    using namespace filesystem;

    try {

        Image x({ 5, 4, 3 }, DataType::uint8, "xyc", ColorType::RGB);
        View<DataType::uint8> y(x);
        y({ 0,0,0 }) = 15; y({ 1,0,0 }) = 16; y({ 2,0,0 }) = 17; y({ 3,0,0 }) = 18; y({ 4,0,0 }) = 19;
        y({ 0,1,0 }) = 25; y({ 1,1,0 }) = 26; y({ 2,1,0 }) = 27; y({ 3,1,0 }) = 28; y({ 4,1,0 }) = 29;
        y({ 0,2,0 }) = 35; y({ 1,2,0 }) = 36; y({ 2,2,0 }) = 37; y({ 3,2,0 }) = 38; y({ 4,2,0 }) = 39;
        y({ 0,3,0 }) = 45; y({ 1,3,0 }) = 46; y({ 2,3,0 }) = 47; y({ 3,3,0 }) = 48; y({ 4,3,0 }) = 49;

        y({ 0,0,1 }) = 17; y({ 1,0,1 }) = 16; y({ 2,0,1 }) = 10; y({ 3,0,1 }) = 17; y({ 4,0,1 }) = 19;
        y({ 0,1,1 }) = 27; y({ 1,1,1 }) = 26; y({ 2,1,1 }) = 20; y({ 3,1,1 }) = 27; y({ 4,1,1 }) = 29;
        y({ 0,2,1 }) = 37; y({ 1,2,1 }) = 36; y({ 2,2,1 }) = 30; y({ 3,2,1 }) = 37; y({ 4,2,1 }) = 39;
        y({ 0,3,1 }) = 47; y({ 1,3,1 }) = 46; y({ 2,3,1 }) = 40; y({ 3,3,1 }) = 47; y({ 4,3,1 }) = 49;

        y({ 0,0,2 }) = 15; y({ 1,0,2 }) = 17; y({ 2,0,2 }) = 17; y({ 3,0,2 }) = 18; y({ 4,0,2 }) = 17;
        y({ 0,1,2 }) = 25; y({ 1,1,2 }) = 27; y({ 2,1,2 }) = 27; y({ 3,1,2 }) = 28; y({ 4,1,2 }) = 27;
        y({ 0,2,2 }) = 35; y({ 1,2,2 }) = 37; y({ 2,2,2 }) = 37; y({ 3,2,2 }) = 38; y({ 4,2,2 }) = 37;
        y({ 0,3,2 }) = 45; y({ 1,3,2 }) = 47; y({ 2,3,2 }) = 47; y({ 3,3,2 }) = 48; y({ 4,3,2 }) = 47;

        Image k1;
        RearrangeChannels(x, k1, "cxy");

        Image k2;
        RearrangeChannels(k1, k2, "xyc");

        Image a1;
        CopyImage(x, a1);
        CopyImage(x, a1);
        CopyImage(x, a1, DataType::uint16);
        Image a2;
        CopyImage(x, a2, DataType::int32);
        CopyImage(a2, a1, DataType::float32);
        CopyImage(a1, a2, DataType::int16);

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

        View<DataType::uint8> cropped1(img, { 0, 0, 2 }, { -1, -1, -1 });
        View<DataType::uint8> cropped2(img, { 100, 100, 2 }, { 200, 200, -1 });
        ImWrite("cropped.png", cropped1);
        ImWrite("cropped.png", cropped2);

        Image test1, test2;
        ChangeColorSpace(cropped1, cropped1, ColorType::BGR);
        ChangeColorSpace(cropped2, cropped2, ColorType::BGR);

        Image img1, img2;
        ImRead("../data/Kodak/img0003.png", img1);
        ImRead("../data/Kodak/img0015.png", img2);

        //ImShow(img1);
        //ResizeScale(img1, img1, { 0.3, 0.3 });

        Image img3, img4;
        CopyImage(img1, img3, DataType::uint16);
        CopyImage(img1, img4, DataType::uint16);

        cv::TickMeter tm;

        //tm.reset();
        //tm.start();
        //Add(img3, img2);
        //tm.stop();
        //std::cout << "Elapsed " << tm.getTimeSec() << " s\n";

        //tm.reset();
        //tm.start();
        //Sub(img3, img2);
        //tm.stop();
        //std::cout << "Elapsed " << tm.getTimeSec() << " s\n";

        //Mul(img1, 2);
        //Mul(0.5, img1);

        img1.Create({ 8192, 2304, 1 }, DataType::int8, "xyc", ColorType::GRAY);
        img1.Create({ 3072, 2048, 3 }, DataType::uint8, "xyc", ColorType::BGR);

        Image img5;
        CopyImage(img1, img5, DataType::int16);
        Neg(img5);
        Neg(img1);

        Div(img1, 4);
        Div(200, img1);

        Add(img1, 100, false);
        Add(-100, img1, false);

        //Sub(img1, 100);
        Sub(100, Sub(img1, 100));


        /*Image mask(img3.dims_, DataType::float32, img3.channels_, img3.colortype_);
        ContiguousViewXYC<DataType::float32> vmask(mask);
        auto radius = float(std::min(vmask.width() / 2, vmask.height() / 2));
        int cx = vmask.width() / 2;
        int cy = vmask.height() / 2;
        for (int y = 0; y < vmask.height(); ++y) {
            int ry = y - cy;
            for (int x = 0; x < vmask.width(); ++x) {
                int rx = x - cx;
                float rd = static_cast<float>(sqrt(rx * rx + ry * ry) / radius);
                vmask(x, y, 0) = vmask(x, y, 1) = vmask(x, y, 2) = std::max(0.f, 1 - rd);
            }
        }

        Mul(img3, vmask);

        Mul(img1, 4, false);

        Sum(img, 100.0);
        Mul(img, 0.5);*/

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
                auto& p = *i;
                auto& p1 = *i1;
                auto& p2 = *i2;
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
    catch (const std::runtime_error& e) {
        std::cerr << "Runtime error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}