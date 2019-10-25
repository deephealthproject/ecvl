#ifndef ECVL_IMGCODECS_H_
#define ECVL_IMGCODECS_H_

#include <filesystem>
#include <string>

#include "image.h"

namespace ecvl {

enum class ImageFormat {
    DEFAULT,
    NIFTI,
#ifdef ECVL_WITH_DICOM
    DICOM,
#endif
};

/** @brief Loads an image from a file.

The function ImRead loads an image from the specified file. If the image cannot
be read for any reason, the function creates an empty Image and returns false.

@anchor imread_path

@param[in] filename A std::filesystem::path identifying the file name.
@param[out] dst Image in which data will be stored.

@return true if the image is correctly read, false otherwise.
*/
bool ImRead(const std::filesystem::path& filename, Image& dst, ImageFormat f = ImageFormat::DEFAULT);


/** @brief Loads a multi-page image from a file.

The function ImReadMulti loads a multi-page image from the specified file. If the image cannot
be read for any reason, the function creates an empty Image and returns false.

@param[in] filename A std::string identifying the file name. In order to be platform
independent consider to use @ref imreadmulti_path "ImReadMulti(const filesystem::path& filename, Image& dst)" .
@param[out] dst Image in which data will be stored.

@return true if the image is correctly read, false otherwise.
*/
bool ImReadMulti(const std::string& filename, Image& dst);

/** @overload

This variant of ImReadMulti is platform independent.

@anchor imreadmulti_path

@param[in] filename A std::filesystem::path identifying the file name.
@param[out] dst Image in which data will be stored.

@return true if the image is correctly read, false otherwise.
*/
bool ImReadMulti(const std::filesystem::path& filename, Image& dst);


/** @brief Saves an image into a specified file.

The function ImWrite saves the input image into a specified file. The image format is chosen based on the 
filename extension. The following sample shows how to create a BGR image and save it to the PNG file "test.png":
@include "../snippets/imgcodecs_imwrite.cpp"

@anchor imwrite_path

@param[in] filename A std::filesystem::path identifying the output file name.
@param[in] src Image to be saved.

@return true if the image is correctly written, false otherwise.
*/
bool ImWrite(const std::filesystem::path& filename, const Image& src, ImageFormat f = ImageFormat::DEFAULT);

} // namespace ecvl

#endif // !ECVL_IMGCODECS_H_

