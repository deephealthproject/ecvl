/*
* ECVL - European Computer Vision Library
* Version: 1.0.3
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include <iostream>
#include <vector>
#include "ecvl/core.h"

using namespace ecvl;
using namespace std;

int main()
{
    // Open an Image
    Image origin;
    if (!ImRead("../examples/data/img0003.png", origin)) {
        return EXIT_FAILURE;
    }

    // Convert input Image to grayscale
    Image gray;
    ChangeColorSpace(origin, gray, ColorType::GRAY);

    // Calculate Otsu threshold value and apply it to the input gray-level Image
    Image otsu;
    int thresh = OtsuThreshold(gray);
    Threshold(gray, otsu, thresh, 255);

    // Calculate Otsu multi-threshold values (2) and apply them to the input 
    // gray-level Image
    Image multi_otsu;
    vector<int> multi_thresh = OtsuMultiThreshold(gray, 2);
    MultiThreshold(gray, multi_otsu, multi_thresh);

    return EXIT_SUCCESS;
}