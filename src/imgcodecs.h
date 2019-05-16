#ifndef ECVL_IMGCODECS_H_
#define ECVL_IMGCODECS_H_

#include <string>

#include "core.h"
#include "filesystem.h"

namespace ecvl {

/** @brief Brief description of the function/procedure.

@anchor value -> to set an invisible link that can be referred to inside the documentation using @ref value command

Complete description of the function/procedure

@note Here you can write special notes that will be displayed differently inside the final documentation (yellow bar on the left)

@param[in] m Description starting with capital letter
@param[out]
@param[in,out]

@return Description of the return value, None if void.
*/
bool ImRead(const std::string& filename, Image& dst);

/** @brief Brief description of the function/procedure.

@anchor value -> to set an invisible link that can be referred to inside the documentation using @ref value command

Complete description of the function/procedure

@note Here you can write special notes that will be displayed differently inside the final documentation (yellow bar on the left)

@param[in] m Description starting with capital letter
@param[out]
@param[in,out]

@return Description of the return value, None if void.
*/
bool ImRead(const filesystem::path& filename, Image& dst);

/** @brief Brief description of the function/procedure.

@anchor value -> to set an invisible link that can be referred to inside the documentation using @ref value command

Complete description of the function/procedure

@note Here you can write special notes that will be displayed differently inside the final documentation (yellow bar on the left)

@param[in] m Description starting with capital letter
@param[out]
@param[in,out]

@return Description of the return value, None if void.
*/
bool ImWrite(const std::string& filename, const Image& src);

/** @brief Brief description of the function/procedure.

@anchor value -> to set an invisible link that can be referred to inside the documentation using @ref value command

Complete description of the function/procedure

@note Here you can write special notes that will be displayed differently inside the final documentation (yellow bar on the left)

@param[in] m Description starting with capital letter
@param[out]
@param[in,out]

@return Description of the return value, None if void.
*/
bool ImWrite(const filesystem::path& filename, const Image& src);

} // namespace ecvl

#endif // !ECVL_IMGCODECS_H_

