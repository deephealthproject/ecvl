/*
* ECVL - European Computer Vision Library
* Version: 0.2.1
* copyright (c) 2020, UniversitÓ degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include "ecvl/core/imgcodecs.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ecvl/core/filesystem.h"
#include "ecvl/core/support_nifti.h"
#include "ecvl/core/support_opencv.h"

#ifdef ECVL_WITH_DICOM
#include "ecvl/core/support_dcmtk.h"
#endif

using namespace ecvl::filesystem;

namespace ecvl
{
bool ImRead(const path& filename, Image& dst, ImReadMode flags)
{
    if (exists(filename)) {
        dst = MatToImage(cv::imread(filename.string(), (int)flags));
        if (dst.IsEmpty()) {
            // NIFTI
            NiftiRead(filename, dst);
        }
#ifdef ECVL_WITH_DICOM
        if (dst.IsEmpty()) {
            // DICOM
            DicomRead(filename, dst);
        }
#endif

        return !dst.IsEmpty();
    }
    else {
        std::cerr << ECVL_ERROR_MSG "File " << filename << " does not exist" << std::endl;
        ECVL_ERROR_FILE_DOES_NOT_EXIST
    }
}

bool ImReadMulti(const path& filename, Image& dst)
{
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