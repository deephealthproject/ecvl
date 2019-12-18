#include "ecvl/core.h"

#include <iostream>

using namespace std;
using namespace ecvl;

int main()
{
    // Create BGR Image
    Image img({ 500, 500, 3 }, DataType::uint8, "xyc", ColorType::BGR);

    // Populate Image with pseudo-random data
    for (int r = 0; r < img.dims_[1]; ++r) {
        for (int c = 0; c < img.dims_[0]; ++c) {
            *img.Ptr({ c, r, 0 }) = 255;
            *img.Ptr({ c, r, 1 }) = (r / 2) % 255;
            *img.Ptr({ c, r, 2 }) = (r / 2) % 255;
        }
    }

    ImWrite("./test.png", img);

    if (!ImRead("./test.png", img)) {
        return EXIT_FAILURE;
    }
    cout << "Successfully read a color image" << endl;

    if (!ImRead("./test.png", img, ImReadMode::GRAYSCALE)) {
        return EXIT_FAILURE;
    }
    cout << "Successfully read a grayscale image" << endl;

    return EXIT_SUCCESS;
}