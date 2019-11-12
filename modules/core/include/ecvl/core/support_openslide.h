#ifndef SUPPORT_OPENSLIDE_H_
#define SUPPORT_OPENSLIDE_H_

#include "ecvl/core/image.h"
#include <filesystem>

namespace ecvl {

/** @brief Loads an image from a HAMAMATSU file.

Loads an image from the specified HAMAMATSU file. If the image cannot
be read for any reason, the function creates an empty Image and returns false.

@anchor hamamatsuread_path

@param[in] filename A filesystem::path identifying the file name.
@param[out] dst Image in which data will be stored.
@param[in] level Image level to be extracted.
@param[in] dims std::vector containing { x, y, w, h }.
            x and y are the top left x-coordinate and y-coordinate, in the level 0 reference frame.
            w and h are the width and height of the region.

@return true if the image is correctly read, false otherwise.
*/
extern bool HamamatsuRead(const std::filesystem::path& filename, Image& dst, const int level, const std::vector<int>& dims);

} // namespace ecvl

#endif // SUPPORT_OPENSLIDE_H_
