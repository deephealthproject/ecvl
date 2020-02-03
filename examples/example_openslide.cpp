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

#include <filesystem>
#include <iostream>
#include "ecvl/core.h"

using namespace ecvl;
using namespace std;

int main()
{
    filesystem::path filename("../examples/data/hamamatsu/test3-DAPI 2 (387).ndpi");
    Image img;
    vector<array<int, 2>> levels;

    // Get dimensions of each level
    if (!OpenSlideGetLevels(filename, levels)) {
        return EXIT_FAILURE;
    }

    // For each level extracts a region of dimensions equal to those of the last level, starting from coordinates (0,0)
    for (int i = 0; i < levels.size(); ++i) {
        vector<int> dims = {
            0, // x
            0, // y
            levels[levels.size() - 1][0],  // w
            levels[levels.size() - 1][1]   // h
        };

        // Read an Hamamatsu file
        if (!OpenSlideRead(filename, img, i, dims)) {
            return EXIT_FAILURE;
        }
        ImWrite("hamamatsu_level_" + to_string(i) + ".png", img);
        cout << "Writing 'hamamatsu_level_" << i << ".png'\n";
    }

    filename = "../examples/data/hamamatsu/test3-FITC 2 (485).ndpi";
    // Image level to be extracted
    int level = 0;
    // Set the RegionOfInterest informations
    vector<int> dims = {
        1000,  // x
        1000, // y
        500,  // w
        500   // h
    };
    if (!OpenSlideRead(filename, img, level, dims)) {
        return EXIT_FAILURE;
    }
    ImWrite("hamamatsu_2.png", img);
    cout << "Writing 'hamamatsu_2.png'\n";

    return EXIT_SUCCESS;
}