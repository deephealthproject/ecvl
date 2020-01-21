/*
* ECVL - European Computer Vision Library
* Version: 0.1
* copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include "ecvl/core.h"

using namespace std;
using namespace ecvl;

int main()
{
    // Loads an existing image
    Image img;
    if (!ImRead("../examples/data/test.jpg", img)) {
        return EXIT_FAILURE;
    }

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
    ImWrite("example_core_iterators.png", img);

    return EXIT_SUCCESS;
}