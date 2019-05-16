#ifndef ECVL_IMGCODECS_H_
#define ECVL_IMGCODECS_H_

#include <string>

#include "core.h"
#include "filesystem.h"

namespace ecvl {

/** @brief Loads an image from a file.

The function ImRead loads an image from the specified file. If the image cannot
be read for any reason, the function creates an empty Image and returns false.

@param[in] filename A std::string identifying the file name. In order to be platform
independent consider to use @ref imread_path.
@param[out] dst Image in which data will be stored.

@return true if the image is correctly read, false otherwise.
*/
bool ImRead(const std::string& filename, Image& dst);

/** @overload

This variant of ImRead is platform independent.

@anchor imread_path

@param[in] filename A filesystem::path identifying the file name.
@param[out] dst Image in which data will be stored.

@return true if the image is correctly read, false otherwise.
*/
bool ImRead(const filesystem::path& filename, Image& dst);

/** @brief Saves an image into a specified file.

The function ImWrite saves the input image into a specified file. The image format is
chosen based on the filename extension.

The following sample shows how to create a BGR image and save it to the PNG file "test.png".
@include ../snippets/imgcodecs_imwrite.cpp

@param[in] filename A std::string identifying the output file name. In order to be platform
independent consider to use @ref imwrite_path.
@param[in] src Image to be saved.

@return true if the image is correctly write, false otherwise.
*/
bool ImWrite(const std::string& filename, const Image& src);

/** @overload

This variant of ImWrite is platform independent.

@anchor imwrite_path

@param[in] filename A filesystem::path identifying the output file name. 
@param[in] src Image to be saved.

@return true if the image is correctly write, false otherwise.
*/
bool ImWrite(const filesystem::path& filename, const Image& src);

} // namespace ecvl

#endif // !ECVL_IMGCODECS_H_

