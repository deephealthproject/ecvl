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

#ifndef ECVL_SUPPORT_OPENCV_H_
#define ECVL_SUPPORT_OPENCV_H_

#include "image.h"

namespace ecvl {

/** @brief Convert a cv::Mat into an ecvl::Image.

@param[in] m Input OpenCV Mat.
@param[in] ctype Input ColorType of the returned ecvl::Image. If not provided the ColorType is guessed from OpenCV Mat.
@param[in] dst_channels Output Image channels order. Default value is empty string, meaning the OpenCV Mat channels order is preserved.

@return ECVL image.
*/
ecvl::Image MatToImage(const cv::Mat& m, ColorType ctype = ColorType::none, const std::string& dst_channels = "");

/** @brief Convert a std::vector<cv::Mat> into an ecvl::Image.

@param[in] v Input std::vector of OpenCV Mat.

@return ECVL image.
*/
ecvl::Image MatVecToImage(const std::vector<cv::Mat>& v);

/** @brief Convert an ECVL Image into OpenCV Mat.

@param[in] img Input ECVL Image.

@return Output OpenCV Mat.
*/
cv::Mat ImageToMat(const Image& img);

} // namespace ecvl 

#endif // ECVL_SUPPORT_OPENCV_H_