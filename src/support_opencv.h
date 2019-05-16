#pragma once

#include "core.h"

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
ecvl::Image MatToImage(const cv::Mat& m);

/** @brief Convert an ECVL Image into OpenCV Mat.

@anchor value -> to set an invisible link that can be referred to inside the documentation using @ref value command

Complete description of the function/procedure

@note Here you can write special notes that will be displayed differently inside the final documentation (yellow bar on the left)

@param[in] m Description starting with capital letter
@param[out]
@param[in,out]

@return Description of the return value, None if void.
*/
cv::Mat ImageToMat(const Image& img);

} // namespace ecvl 