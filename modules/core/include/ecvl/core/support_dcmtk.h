#ifndef SUPPORT_DCMTK_H_
#define SUPPORT_DCMTK_H_

#include "ecvl/core/image.h"
#include <filesystem>

namespace ecvl {

/** @brief Loads an image from a DICOM file.

Loads an image from the specified DICOM file. If the image cannot
be read for any reason, the function creates an empty Image and returns false.

@anchor dicomread_path

@param[in] filename A filesystem::path identifying the file name.
@param[out] dst Image in which data will be stored.

@return true if the image is correctly read, false otherwise.
*/
extern bool DicomRead(const std::filesystem::path& filename, Image& dst);

/** @brief Saves an image into a specified DICOM file.

The function DicomWrite saves the input image into a specified file, with the DICOM format.

@anchor dicomwrite_path

@param[in] filename A filesystem::path identifying the output file name.
@param[in] src Image to be saved.

@return true if the image is correctly written, false otherwise.
*/
extern bool DicomWrite(const std::filesystem::path& filename, const Image& src);

/** @example example_nifti_dicom.cpp
 Nifti and Dicom support example.
*/
} // namespace ecvl

#endif // SUPPORT_DCMTK_H_
