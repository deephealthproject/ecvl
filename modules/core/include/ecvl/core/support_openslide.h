#ifndef SUPPORT_OPENSLIDE_H_
#define SUPPORT_OPENSLIDE_H_

#include "ecvl/core/image.h"
#include <filesystem>

namespace ecvl {

/** @brief Loads a region of a whole-slide image file.

Loads a region from the specified whole-slide image file. Supported formats are those supported by OpenSlide library.
If the region cannot be read for any reason, the function creates an empty Image and returns false.

@anchor openslideread_path

@param[in] filename A filesystem::path identifying the file name.
@param[out] dst Image in which data will be stored.
@param[in] level Image level to be extracted.
@param[in] dims std::vector containing { x, y, w, h }.
            x and y are the top left x-coordinate and y-coordinate, in the level 0 reference frame.
            w and h are the width and height of the region.

@return true if the image is correctly read, false otherwise.
*/
extern bool OpenSlideRead(const std::filesystem::path& filename, Image& dst, const int level, const std::vector<int>& dims);

/** @brief Get width and height for each level of a whole-slide image. 

@param[in] filename A filesystem::path identifying the file name.
@param[out] levels A std::vector of array containing two elements, width and height respectively.
            levels[k] are the dimensions of level k.
*/
void OpenSlideGetLevels(const std::filesystem::path& filename, std::vector<std::array<int, 2>>& levels);

/** @example example_openslide.cpp
 Openslide support example.
*/
} // namespace ecvl

#endif // SUPPORT_OPENSLIDE_H_
