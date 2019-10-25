#include <iostream>
#include "ecvl/core.h"

using namespace ecvl;
using namespace std;

int main(void)
{
    // Create an empty Image 
    Image x({ 5, 4, 3 }, DataType::uint8, "xyc", ColorType::RGB);
    
    // Create a view from the previous Image and fill them with data
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

    // Change channels order
    Image k;
    cout << "Executing RearrangeChannels" << endl;
    RearrangeChannels(x, k, "cxy");

    // Change DataType of an Image (src and dst cannot be the same image!)
    cout << "Executing CopyImage" << endl;
    CopyImage(x, k, DataType::uint16);

    // Read ECVL image from file
    Image img;
    if (!ImRead("../data/test.jpg", img)) {
        return EXIT_FAILURE;
    }

    // View of an Image which starts at coordinates { 0, 0, 2 } until the end of the channels (because sizes are negative)
    // cropped1 contains a 200x200 square of the last color channel of img
    View<DataType::uint8> cropped1(img, { 0, 0, 2 }, { 200, 200, -1 });

    // Change the color space of the Image
    cout << "Executing ChangeColorSpace" << endl;
    ChangeColorSpace(cropped1, cropped1, ColorType::BGR);

    // Also a View can be saved with ImWrite
    ImWrite("cropped.png", cropped1);

    //Create a thumbnail 32x32 from an Image
    Image img1, img2, img3, img4;
    ImRead("../data/Kodak/img0003.png", img1);

    cout << "Create a thumbnail 32x32 from an Image" << endl;
    CopyImage(View<DataType::uint8>(img1, { 0,0,0 }, { -1,-1,1 }), img2);
    img3.Create({ 32,32,1 }, DataType::uint8, "xyc", ColorType::GRAY);
    std::vector<uint32_t> accum(32);
    std::vector<uint32_t> count(32);
    uint32_t* paccum, *pcount;
    ContiguousViewXYC<DataType::uint8> v3(img2);
    auto i = v3.Begin();
    auto pout = img3.data_;
    for (int dr = 0, sr = 0, w = v3.width(), h = v3.height(); sr < h; ++dr) {
        memset(paccum = accum.data(), 0, 32 * 4);
        memset(pcount = count.data(), 0, 32 * 4);
        while (sr * 32 / h == dr) {
            paccum = accum.data();
            pcount = count.data();
            for (int dc = 0, sc = 0; sc < w; ++sc) {
                *paccum += *i;
                *pcount += 1;
                ++i;
                if (sc * 32 / w > dc) {
                    ++dc;
                    ++paccum;
                    ++pcount;
                }
            }
            sr++;
        }
        std::transform(begin(accum), end(accum), begin(count), pout, std::divides<uint32_t>());
        pout += 32;
    }
    ImWrite("thumbnail.jpg", img3);

    // Create an empty Image
    cout << "Create an empty Image" << endl;
    img1.Create({ 3072, 2048, 3 }, DataType::uint8, "xyc", ColorType::BGR);

    ImRead("../data/Kodak/img0015.png", img2);
    CopyImage(img1, img3, DataType::uint16);
    CopyImage(img1, img4, DataType::int32);

    Image dst1, dst2, dst3, dst4, dst5;
    cout << "Executing arithmetic functions" << endl;
    // Test add functions
    Add(img1, img2, dst1);
    Add(img1, 100, dst2);
    Add(200, img1, dst2);

    // Test sub functions
    Sub(img1, img2, dst1);
    Sub(img1, 100, dst2);
    Sub(300, img1, dst2);

    // Test mul functions
    Mul(img4, 256 * 256 * 128, dst3);
    Mul(dst3, 3, dst4);
    Mul(dst3, 3, dst5, false);
    Mul(img4, img2, dst3);
    Mul(img4, 256, dst3);
    Mul(512, img4, dst3);

    // Test div functions
    Div(img1, img2, dst1);
    Div(img2, img1, dst2);
    Div(img1, 2, dst2);
    Div(200, img1, dst3);

    // Create a black mask with a blurred white circle in the center
    cout << "Create a mask" << endl;
    Image mask(img3.dims_, DataType::float32, img3.channels_, img3.colortype_);
    ContiguousViewXYC<DataType::float32> vmask(mask);
    auto radius = float(std::min(vmask.width() / 2, vmask.height() / 2));
    int cx = vmask.width() / 2;
    int cy = vmask.height() / 2;
    for (int y = 0; y < vmask.height(); ++y) {
        int ry = y - cy;
        for (int x = 0; x < vmask.width(); ++x) {
            int rx = x - cx;
            auto rd = static_cast<float>(sqrt(rx * rx + ry * ry) / radius);
            vmask(x, y, 0) = vmask(x, y, 1) = vmask(x, y, 2) = std::max(0.f, 1 - rd);
        }
    }
    Mul(mask, 255, mask);
    ImWrite("mask.jpg", mask);

    return EXIT_SUCCESS;
}