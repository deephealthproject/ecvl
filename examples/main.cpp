#include <iostream>

#include "../src/support_opencv.h"
#include "../src/imgcodecs.h"
#include "../src/filesystem.h"

using namespace ecvl;
using namespace filesystem;

int main(void)
{
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

    cv::Mat m = cv::imread(path("../data/test.jpg").string());
    Image img = MatToImage(m);
    m = ImageToMat(img);
    
    //Image j = ecvl::ImRead(path("test.jpg"));
    return 0;
}