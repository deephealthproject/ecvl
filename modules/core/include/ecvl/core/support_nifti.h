#ifndef ECVL_SUPPORT_NIFTI_H_
#define ECVL_SUPPORT_NIFTI_H_

#include "image.h"
#include "filesystem.h"

namespace ecvl {

/** @brief Loads a nifti image from a file.

The function NiftiRead loads an image from the specified nifti file. If the image cannot
be read for any reason, the function creates an empty Image and returns false.

@param[in] filename A std::string identifying the file name. In order to be platform
independent consider to use @ref niftiread_path "NiftiRead(const filesystem::path& filename, Image& dst)" .
@param[out] dst Image in which data will be stored.

@return true if the image is correctly read, false otherwise.
*/
bool NiftiRead(const std::string& filename, Image& dst);

/** @overload

This variant of NiftiRead is platform independent.

@anchor niftiread_path

@param[in] filename A filesystem::path identifying the file name.
@param[out] dst Image in which data will be stored.

@return true if the image is correctly read, false otherwise.
*/
bool NiftiRead(const filesystem::path& filename, Image& dst);

/** @brief Saves an image into a specified nifti file.

The function NiftiWrite saves the input image into a specified file, with the NIfTI-1 format.

@param[in] filename A std::string identifying the output file name. In order to be platform
independent consider to use @ref niftiwrite_path "NiftiWrite(const filesystem::path& filename, const Image& src)".
@param[in] src Image to be saved.

@return true if the image is correctly written, false otherwise.
*/
bool NiftiWrite(const std::string& filename, const Image& src);

/** @overload

This variant of NiftiWrite is platform independent.

@anchor niftiwrite_path

@param[in] filename A filesystem::path identifying the output file name.
@param[in] src Image to be saved.

@return true if the image is correctly written, false otherwise.
*/
bool NiftiWrite(const filesystem::path& filename, const Image& src);

} // namespace ecvl 

#endif // ECVL_SUPPORT_NIFTI_H_