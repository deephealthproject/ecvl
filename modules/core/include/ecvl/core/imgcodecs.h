#ifndef ECVL_IMGCODECS_H_
#define ECVL_IMGCODECS_H_

#include <filesystem>
#include <string>

#include "image.h"

namespace ecvl {

/**  @brief ImageFormat is an enum class which defines
the images format to employ.

 @anchor ImageFormat
 */
enum class ImageFormat {
    DEFAULT, /**< Any common format */
    NIFTI,   /**< NIfTI data format */
#ifdef ECVL_WITH_DICOM
    DICOM,   /**< DICOM data format */
#endif
};

/** @brief Loads an image from a file.

The function ImRead loads an image from the specified file. If the image cannot
be read for any reason, the function creates an empty Image and returns false.

@anchor imread_path

@param[in] filename A std::filesystem::path identifying the file name.
@param[out] dst Image in which data will be stored.
@param[in] f A ImageFormat indicating the image format to read.

@return true if the image is correctly read, false otherwise.
*/
bool ImRead(const std::filesystem::path& filename, Image& dst, ImageFormat f = ImageFormat::DEFAULT);


/** @brief Loads a multi-page image from a file.

The function ImReadMulti loads a multi-page image from the specified file. If the image cannot
be read for any reason, the function creates an empty Image and returns false.

@param[in] filename A std::string identifying the file name.
@param[out] dst Image in which data will be stored.

@return true if the image is correctly read, false otherwise.
*/
bool ImReadMulti(const std::filesystem::path& filename, Image& dst);

/** @brief Saves an image into a specified file.

The function ImWrite saves the input image into a specified file. The image format is chosen based on the
filename extension. The following sample shows how to create a BGR image and save it to the PNG file "test.png":
@include "../examples/example_imgcodecs.cpp"

@anchor imwrite_path

@param[in] filename A std::filesystem::path identifying the output file name.
@param[in] src Image to be saved.
@param[in] f A ImageFormat indicating the image format to write.

@return true if the image is correctly written, false otherwise.
*/
bool ImWrite(const std::filesystem::path& filename, const Image& src, ImageFormat f = ImageFormat::DEFAULT);


/** @example example_imgcodecs.cpp
 An example imgcodecs functionalities.
*/

} // namespace ecvl

#endif // !ECVL_IMGCODECS_H_

