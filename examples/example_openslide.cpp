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
#include "ecvl/core.h"
#include "ecvl/core/filesystem.h"

using namespace ecvl;
using namespace std;

using ecvl::filesystem::path;

void Performance()
{
    path filename("../examples/data/hamamatsu/test3-DAPI 2 (387).ndpi");
    OpenSlideImage os_img(filename);
    Image img;
    int n_test = 100;
    vector<double> timings_ecvl(n_test);
    vector<double> timings_openslide(n_test);

    int n_patches = 100;
    int patch_size = 256;

    cv::TickMeter tm;

    for (int i = 0; i < n_test; ++i) {
        tm.reset();
        tm.start();
        for (int j = 0; j < n_patches * patch_size; j += patch_size) {
            os_img.ReadRegion(img, 0, { j, j, patch_size, patch_size });
        }
        tm.stop();
        timings_ecvl[i] = tm.getTimeMilli();
        //cout << "Elapsed time ECVL: " << timings_ecvl[i] << endl;

        openslide_t* osr = openslide_open(filename.string().c_str());
        vector<uint32_t> d(sizeof(uint32_t) * patch_size * patch_size);
        tm.reset();
        tm.start();
        for (int j = 0; j < n_patches * patch_size; j += patch_size) {
            openslide_read_region(osr, d.data(), j, j, 0, patch_size, patch_size);
        }
        tm.stop();
        timings_openslide[i] = tm.getTimeMilli();
        //cout << "Elapsed time OpenSlide: " << timings_openslide[i] << endl;
    }

    auto t_ecvl = accumulate(timings_ecvl.begin(), timings_ecvl.end(), 0.) / n_test;
    auto t_openslide = accumulate(timings_openslide.begin(), timings_openslide.end(), 0.) / n_test;

    cout << "Elapsed mean time ECVL: " << t_ecvl << endl;
    cout << "Elapsed mean time OpenSlide: " << t_openslide << endl;
}

int main()
{
    filesystem::path filename("../examples/data/hamamatsu/test3-DAPI 2 (387).ndpi");
    OpenSlideImage os_img(filename);
    Image img;
    vector<array<int, 2>> levels;

    // Get number of levels
    int n_levels = os_img.GetLevelCount();

    // Get dimensions of each level
    os_img.GetLevelsDimensions(levels);

    // For each level extracts a region of dimensions equal to those of the last level, starting from coordinates (0,0)
    for (int i = 0; i < levels.size(); ++i) {
        vector<int> dims = {
            0, // x
            0, // y
            levels[levels.size() - 1][0],  // w
            levels[levels.size() - 1][1]   // h
        };

        // Read a region of the Hamamatsu file
        os_img.ReadRegion(img, i, dims);

        // Read metadata from OpenSlide file and store them into an ECVL Image
        os_img.GetProperties(img);

        // All the openslide metadata are stored in strings
        for (auto& p : img.meta_) {
            cout << p.first << " - " << any_cast<string>(p.second.Get()) << endl;
        }

        // save in variables specific metadata
        auto mpp_x = img.GetMeta("openslide.mpp-x");
        auto roi_slide_macro = img.GetMeta("hamamatsu.roi.slide.macro");

        cout << "mpp-x - " << any_cast<string>(mpp_x.Get()) << endl;

        // save levels as png, without format specific metadata
        ImWrite("hamamatsu_level_" + to_string(i) + ".png", img);
        cout << "Writing 'hamamatsu_level_" << i << ".png'\n";
    }

    filename = "../examples/data/hamamatsu/test3-FITC 2 (485).ndpi";
    OpenSlideImage os_img_2(filename);

    // Image level to be extracted
    int level = 0;
    // Set the RegionOfInterest informations
    vector<int> dims = {
        1000,  // x
        1000, // y
        500,  // w
        500   // h
    };

    os_img_2.ReadRegion(img, level, dims);

    ImWrite("hamamatsu_2.png", img);
    cout << "Writing 'hamamatsu_2.png'\n";

    Performance();

    return EXIT_SUCCESS;
}