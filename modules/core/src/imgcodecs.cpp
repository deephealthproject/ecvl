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

#include "ecvl/core/imgcodecs.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ecvl/core/support_opencv.h"

using namespace std::filesystem;

namespace ecvl {

bool ImRead(const path& filename, Image& dst, ImReadMode flags)
{
    dst = MatToImage(cv::imread(filename.string(), (int)flags));
    return !dst.IsEmpty();
}

bool ImReadMulti(const path& filename, Image& dst) {

    std::vector<cv::Mat> v;
    cv::imreadmulti(filename.string(), v);
    dst = MatVecToImage(v);

    return !dst.IsEmpty();
}

bool ImWrite(const path& filename, const Image& src)
{
    return cv::imwrite(filename.string(), ImageToMat(src));
}

} // namespace ecvl