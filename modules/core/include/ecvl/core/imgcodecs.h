/*
* ECVL - European Computer Vision Library
* Version: 1.0.2
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#ifndef ECVL_IMGCODECS_H_
#define ECVL_IMGCODECS_H_

#include "ecvl/core/filesystem.h"
#include "ecvl/core/image.h"

#include <string>

namespace ecvl
{
/** @brief Enum class representing the ECVL ImRead flags.

    @anchor ImReadMode
 */
enum class ImReadMode
{
    UNCHANGED = -1, //!< If set, return the loaded image as is (with alpha channel, otherwise it gets cropped).
    GRAYSCALE = 0,       //!< If set, always convert image to the single channel grayscale image (codec internal conversion).
    COLOR = 1,           //!< If set, always convert image to the 3 channel BGR color image.
    //IMREAD_ANYDEPTH = 2,  //!< If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
    ANYCOLOR = 4,  //!< If set, the image color format is deduced from file format.
};

/** @brief Loads an image from a file.

The function ImRead loads an image from the specified file (jpg, png, dicom or nifti). If the image cannot
be read for any reason, the function creates an empty Image and returns false.

@anchor imread_path

@param[in] filename A std::filesystem::path identifying the file name.
@param[out] dst Image in which data will be stored.
@param[in] flags \ref ImReadMode indicating how to read the image.

@return true if the image is correctly read, false otherwise.
*/
bool ImRead(const ecvl::filesystem::path& filename, Image& dst, ImReadMode flags = ImReadMode::ANYCOLOR);

/**
@brief Loads an image from a buffer in memory. This is an overloaded function, provided for convenience.

The buffer must be a raw encoded image (png, jpg).
If the image cannot be read for any reason, the function creates an empty Image and returns false.

@param[in] buffer A char* identifying the input buffer.
@param[in] size Dimension of the input buffer.
@param[out] dst Image in which data will be stored.
@param[in] flags \ref ImReadMode indicating how to read the image.

@return true if the image is correctly read, false otherwise.
*/
bool ImRead(const char* buffer, const int size, Image& dst, ImReadMode flags = ImReadMode::ANYCOLOR);

/** @brief Loads an image from a buffer in memory. This is an overloaded function, provided for convenience. 

It differs from the above function only because it accepts a std::vector<char> instead of a char*.

@param[in] buffer A std::vector<char> identifying the input buffer.
@param[out] dst Image in which data will be stored.
@param[in] flags \ref ImReadMode indicating how to read the image.

@return true if the image is correctly read, false otherwise.
*/
bool ImRead(const std::vector<char>& buffer, Image& dst, ImReadMode flags = ImReadMode::ANYCOLOR);

/** @brief Loads a multi-page image from a file.

The function ImReadMulti loads a multi-page image from the specified file. If the image cannot
be read for any reason, the function creates an empty Image and returns false.

@param[in] filename A std::string identifying the file name.
@param[out] dst Image in which data will be stored.

@return true if the image is correctly read, false otherwise.
*/
bool ImReadMulti(const ecvl::filesystem::path& filename, Image& dst);

/** @brief Saves an image into a specified file.

The function ImWrite saves the input image into a specified file. The image format is chosen based on the
filename extension. The following sample shows how to create a BGR image and save it to the PNG file "test.png":
@include "../examples/example_imgcodecs.cpp"

@anchor imwrite_path

@param[in] filename A std::filesystem::path identifying the output file name.
@param[in] src Image to be saved.

@return true if the image is correctly written, false otherwise.
*/
bool ImWrite(const ecvl::filesystem::path& filename, const Image& src);

/** @example example_imgcodecs.cpp
 Imgcodecs example.
*/
} // namespace ecvl

#endif // !ECVL_IMGCODECS_H_
