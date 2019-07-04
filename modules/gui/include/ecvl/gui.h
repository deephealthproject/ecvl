#ifndef ECVL_GUI_H_
#define ECVL_GUI_H_

#include <wx/wx.h>
#include "wx/glcanvas.h"
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

void ImShow3D(const Image& img);

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
} // namespace ecvl

#endif // ECVL_GUI_H_