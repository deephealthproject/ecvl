#include "ecvl/core.h"

using namespace std;
using namespace ecvl;

int main()
{
    // Loads an existing image
    Image img;
    ImRead("../data/test.jpg", img);
    
    // Iterates over the image with ContiguosIterator
    // and modifies its pixels values on each channel
    // by increasing their value of 10 (no saturation 
    // involved in the process).
    for (auto i = img.ContiguousBegin<uint8_t>(), e = img.ContiguousEnd<uint8_t>(); i != e; ++i) {
        auto& p = *i;
        p = static_cast<uint8_t>(p + 10);
    }

    // The same results can be obtained with Iterators 
    // (non contiguous), but performance is worse. 
    

    // Write the output image
    ImWrite("./test.png", img);

    return EXIT_SUCCESS;
}