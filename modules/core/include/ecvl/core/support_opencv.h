#pragma once

#include "image.h"

namespace ecvl {

/** @brief Convert a cv::Mat into an ecvl::Image.

@param[in] m Input OpenCV Mat.

@return ECVL image.
*/
ecvl::Image MatToImage(const cv::Mat& m);

/** @brief Convert an ECVL Image into OpenCV Mat.

@param[in] img Input ECVL Image.

@return Output OpenCV Mat.
*/
cv::Mat ImageToMat(const Image& img);

} // namespace ecvl 