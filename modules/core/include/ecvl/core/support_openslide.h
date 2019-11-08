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

@return true if the image is correctly read, false otherwise.
*/
extern bool HamamatsuRead(const std::filesystem::path& filename, Image& dst, int x, int y, int w, int h);

} // namespace ecvl

#endif // SUPPORT_OPENSLIDE_H_
