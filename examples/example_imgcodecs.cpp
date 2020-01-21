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

    ImWrite("example_imgcodecs.png", img);

    if (!ImRead("example_imgcodecs.png", img)) {
        return EXIT_FAILURE;
    }
    cout << "Successfully read a color image" << endl;

    if (!ImRead("example_imgcodecs.png", img, ImReadMode::GRAYSCALE)) {
        return EXIT_FAILURE;
    }
    cout << "Successfully read a grayscale image" << endl;

    return EXIT_SUCCESS;
}