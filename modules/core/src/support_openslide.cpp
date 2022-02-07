/*
* ECVL - European Computer Vision Library
* Version: 1.0.0
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include "ecvl/core/support_openslide.h"

using namespace std;

namespace ecvl
{
void OpenSlideImage::GetLevelsDimensions(std::vector<std::array<int, 2>>& levels)
{
    levels.clear();
    levels.resize(n_levels_);
    long long w, h;
    for (int i = 0; i < n_levels_; ++i) {
        openslide_get_level_dimensions(osr_, i, &w, &h);
        levels[i] = std::array<int, 2>{ static_cast<int>(w), static_cast<int>(h) };
    }
}

void OpenSlideImage::GetLevelDownsamples(std::vector<double>& levels)
{
    levels.clear();
    levels.resize(n_levels_);
    for (int i = 0; i < n_levels_; ++i) {
        levels[i] = openslide_get_level_downsample(osr_, i);
    }
}

int OpenSlideImage::GetBestLevelForDownsample(const double& downsample)
{
    return openslide_get_best_level_for_downsample(osr_, downsample);
}

void OpenSlideImage::GetProperties(Image& dst)
{
    const char* const* prop_names = openslide_get_property_names(osr_);
    int pc = 0;
    while (prop_names[pc] != NULL) {
        string pval = openslide_get_property_value(osr_, prop_names[pc]);
        dst.meta_.insert({ prop_names[pc], MetaData(pval, 0) });
        ++pc;
    }
}

bool OpenSlideImage::ReadRegion(Image& dst, const int level, const std::vector<int>& dims)
{
    const int& x = dims[0];
    const int& y = dims[1];
    const int& w = dims[2];
    const int& h = dims[3];

    std::vector<uint32_t> d(sizeof(uint32_t) * w * h);
    openslide_read_region(osr_, d.data(), x, y, level, w, h);
    dst.Create({ 3, w, h }, DataType::uint8, "cxy", ColorType::RGB);

    uint8_t a, r, g, b;
    uint32_t pixel;
    for (int i = 0, j = 0; i < dst.datasize_; ++j, ++i) {
        pixel = d[j];
        a = pixel >> 24;
        switch (a) {
        case 0:
            r = g = b = 0;
            break;
        case 255:
            r = (pixel >> 16) & 0xff;
            g = (pixel >> 8) & 0xff;
            b = pixel & 0xff;
            break;
        default:
            r = 255 * ((pixel >> 16) & 0xff) / a;
            g = 255 * ((pixel >> 8) & 0xff) / a;
            b = 255 * (pixel & 0xff) / a;
        }
        dst.data_[i] = r;
        dst.data_[++i] = g;
        dst.data_[++i] = b;
    }

    return !dst.IsEmpty();
}
} // namespace ecvl