#include "ecvl/core.h"

using namespace std;
using namespace ecvl;
using namespace filesystem;

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

    ImWrite(path("./test.png"), img);

    return EXIT_SUCCESS;
}