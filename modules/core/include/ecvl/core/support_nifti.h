#ifndef ECVL_SUPPORT_NIFTI_H_
#define ECVL_SUPPORT_NIFTI_H_

#include "image.h"
#include <filesystem>

namespace ecvl {

/** @brief Loads a nifti image from a file.

The function NiftiRead loads an image from the specified nifti file. If the image cannot
be read for any reason, the function creates an empty Image and returns false.

@anchor niftiread_path

@param[in] filename A std::filesystem::path identifying the file name.
@param[out] dst Image in which data will be stored.

@return true if the image is correctly read, false otherwise.
*/
bool NiftiRead(const std::filesystem::path& filename, Image& dst);

/** @brief Saves an image into a specified nifti file.

The function NiftiWrite saves the input image into a specified file, with the NIfTI-1 format.

@anchor niftiwrite_path

@param[in] filename A std::filesystem::path identifying the output file name.
@param[in] src Image to be saved.

@return true if the image is correctly written, false otherwise.
*/
bool NiftiWrite(const std::filesystem::path& filename, const Image& src);
} // namespace ecvl 

#endif // ECVL_SUPPORT_NIFTI_H_