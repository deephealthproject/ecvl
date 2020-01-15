/*
* ECVL - European Computer Vision Library
* Version: 0.1
* copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors: 
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#ifndef ECVL_GUI_H_
#define ECVL_GUI_H_

#include <wx/wx.h>
#include "wx/notebook.h"
#undef _

#include "ecvl/core/image.h"

namespace ecvl {
/** @brief Displays an Image.

The ImShow function instantiates a ShowApp and starts it with a wxEntry() call.
The image is shown with its original size. 

@param[in] img Image to be shown.

*/
void ImShow(const Image& img);

/** @brief Convert an ECVL Image into a wxImage.

@param[in] img Input ECVL Image.

@return wxImage.
*/

wxImage WxFromImg(Image& img);

/** @brief Convert a wxImage into an ECVL Image.

@param[in] wx Input wxImage.

@return ECVL Image.
*/

Image ImgFromWx(const wxImage& wx);

#if defined ECVL_WITH_OPENGL
/** @brief Displays a 3D Image (volume).

The ImShow3D function instantiates a ShowApp and starts it with a wxEntry() call.

@param[in] img Image to be shown.

*/
void ImShow3D(const Image& img);
#endif

/** @example example_ecvl_gui.cpp
 Example of ECVL gui.
*/
} // namespace ecvl

#endif // ECVL_GUI_H_