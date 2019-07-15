#ifndef SUPPORT_DCMTK_H_
#define SUPPORT_DCMTK_H_

#include "ecvl/core/image.h"
#include "ecvl/core/filesystem.h"

namespace ecvl {

/** @brief Loads an image from a DICOM file.

Loads an image from the specified DICOM file. If the image cannot
be read for any reason, the function creates an empty Image and returns false.

@param[in] filename A std::string identifying the file name. In order to be platform
independent consider to use @ref dicomread_path "DicomRead(const filesystem::path& filename, Image& dst)" .
@param[out] dst Image in which data will be stored.

@return true if the image is correctly read, false otherwise.
*/
extern bool DicomRead(const std::string& filename, Image& dst);

/** @overload

This variant of DicomRead is platform independent.

@anchor dicomread_path

@param[in] filename A filesystem::path identifying the file name.
@param[out] dst Image in which data will be stored.

@return true if the image is correctly read, false otherwise.
*/
extern bool DicomRead(const filesystem::path& filename, Image& dst);

} // namespace ecvl

#endif // SUPPORT_DCMTK_H_
