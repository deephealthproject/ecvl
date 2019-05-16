#ifndef ECVL_IMGPROC_H_
#define ECVL_IMGPROC_H_

#include "core.h"
#include "support_opencv.h"

namespace ecvl {

/* Enum class representing the interpolation types.
*  The documentation block cannot be put after the enum!
*/
enum class InterpolationType {
    nearest,    /**< enum value 0 Nearest neighbor interpolation  */
    linear,     /**< enum value 1 Bilinear interpolation */
    area,       /**< enum value 2 Resampling using pixel area relation. 
                     It may be a preferred method for image decimation, 
                     as it gives moire-free results. But when the image 
                     is zoomed, it is similar to the nearest method.*/
    cubic,      /**< enum value 3 Bicubic interpolation  */
    lanczos4    /**< enum value 4 Lanczos interpolation over 8x8 neighborhood */
};

/* @brief Loads an image from a file.

The function ImRead loads an image from the specified file. If the image cannot
be read for any reason, the function creates an empty Image and returns false.

@param[in] filename A std::string identifying the file name. In order to be platform
independent consider to use @ref imread_path.
@param[out] dst Image in which data will be stored.

@return true if the image is correctly read, false otherwise.
*/
static int GetOpenCVInterpolation(InterpolationType interp);

void Resize(const ecvl::Image& src, ecvl::Image& dst, const std::vector<int>& newdims, InterpolationType interp = InterpolationType::linear);

void Flip2D(const ecvl::Image& src, ecvl::Image& dst);

void Mirror2D(const ecvl::Image& src, ecvl::Image& dst);

void Rotate2D(const ecvl::Image& src, ecvl::Image& dst, double angle, const std::vector<double>& center = {}, double scale = 1.0, InterpolationType interp = InterpolationType::linear);

void RotateFullImage2D(const ecvl::Image& src, ecvl::Image& dst, double angle, double scale = 1.0, InterpolationType interp = InterpolationType::linear);

} // namespace ecvl

#endif // ECVL_IMGPROC_H_