/*
* ECVL - European Computer Vision Library
* Version: 0.3.4
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#ifndef SUPPORT_DCMTK_H_
#define SUPPORT_DCMTK_H_

#include "ecvl/core/filesystem.h"
#include "ecvl/core/image.h"

namespace ecvl
{
class OverlayMetaData : public MetaData
{
    ecvl::Image overlay_;

public:

    OverlayMetaData(const ecvl::Image& overlay) : overlay_(overlay) {}
    OverlayMetaData(ecvl::Image&& overlay) = delete;    // always copy, so that memory is contiguous
    virtual bool Query(const std::string& name, std::string& value) const override;
};

/** @brief Loads an image from a DICOM file.

Loads an image from the specified DICOM file. If the image cannot
be read for any reason, the function creates an empty Image and returns false.

@anchor dicomread_path

@param[in] filename A filesystem::path identifying the file name.
@param[out] dst Image in which data will be stored.

@return true if the image is correctly read, false otherwise.
*/
extern bool DicomRead(const ecvl::filesystem::path& filename, Image& dst);

/** @brief Saves an image into a specified DICOM file.

The function DicomWrite saves the input image into a specified file, with the DICOM format.

@anchor dicomwrite_path

@param[in] filename A filesystem::path identifying the output file name.
@param[in] src Image to be saved.

@return true if the image is correctly written, false otherwise.
*/
extern bool DicomWrite(const ecvl::filesystem::path& filename, const Image& src);

struct InitDCMTK
{
    InitDCMTK();
    ~InitDCMTK();
};

/** @example example_nifti_dicom.cpp
 Nifti and Dicom support example.
*/
} // namespace ecvl

#endif // SUPPORT_DCMTK_H_
