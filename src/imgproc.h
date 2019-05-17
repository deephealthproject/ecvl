#ifndef ECVL_IMGPROC_H_
#define ECVL_IMGPROC_H_

#include "core.h"
#include "support_opencv.h"

namespace ecvl {

/* @anchor InterpolationType

   Enum class representing the ECVL interpolation types.
*  
*/
enum class InterpolationType {
    nearest,    /**< Nearest neighbor interpolation  */
    linear,     /**< Bilinear interpolation */
    area,       /**< Resampling using pixel area relation. 
                     It may be a preferred method for image decimation, 
                     as it gives moire-free results. But when the image 
                     is zoomed, it is similar to the nearest method.*/
    cubic,      /**< Bicubic interpolation  */
    lanczos4    /**< Lanczos interpolation over 8x8 neighborhood */
};

/* @brief Resizes an Image to a new dimension

The function resizes Image src and outputs the result in dst.

@param[in] src The input Image.
@param[out] dst The output resized Image.
@param[in] newdims std::vector<int> that specifies the new size of each dimension.
            The vector size must match the src Image dimentions, excluding the color channel
@param[in] interp InterpolationType to be used. See @ref InterpolationType.

*/
void ResizeDim(const ecvl::Image& src, ecvl::Image& dst, const std::vector<int>& newdims, InterpolationType interp = InterpolationType::linear);

/* @brief Resizes an Image by scaling the dimentions to a given scale factor

The function resizes Image src and outputs the result in dst.

@param[in] src The input Image.
@param[out] dst The output resized Image.
@param[in] scales std::vector<double> that specifies the scale to apply to each dimension.
            The vector size must match the src Image dimentions, excluding the color channel.
@param[in] interp InterpolationType to be used. See @ref InterpolationType.

*/
void ResizeScale(const ecvl::Image& src, ecvl::Image& dst, const std::vector<double>& scales, InterpolationType interp = InterpolationType::linear);

/* @brief Flips an Image

The Flip2D procedure vertically flips an Image.

@param[in] src The input Image.
@param[out] dst The output flipped Image.

*/
void Flip2D(const ecvl::Image& src, ecvl::Image& dst);

/* @brief Mirrors an Image

The Mirror2D procedure horizontally flips an Image.

@param[in] src The input Image.
@param[out] dst The output mirrored Image.

*/
void Mirror2D(const ecvl::Image& src, ecvl::Image& dst);

/* @brief Rotates an Image

The Rotate2D procedure rotates an Image of a given angle (expressed in degrees) in a clockwise manner, with respect to a 
given center. The value of unknown pixels in the output Image are set to 0. The output Image is guaranteed to have the same 
dimensions as the input one. An optional scale parameter can be provided: this won't change the output Image size, but the 
image is scaled during rotation. Different interpolation types are available, see @ref InterpolationTypes.

@param[in] src The input Image.
@param[out] dst The output rotated Image.
@param[in] angle The rotation angle in degrees.
@param[in] center A std::vector<double> representing the coordinates of the rotation center. 
            If empty, the center of the image is used.
@param[in] scale Optional scaling factor.
@param[in] interp Interpolation type used. Default is InterpolationType::linear.

*/
void Rotate2D(const ecvl::Image& src, ecvl::Image& dst, double angle, const std::vector<double>& center = {}, double scale = 1.0, InterpolationType interp = InterpolationType::linear);

/* @brief Rotates an Image resizing the output accordingly.

The RotateFullImage2D procedure rotates an Image of a given angle (expressed in degrees) in a clockwise manner. 
The value of unknown pixels in the output Image are set to 0. The output Image is guaranteed to contain all the pixels 
of the rotated image. Thus, its dimensions can be different from those of the input. 
An optional scale parameter can be provided. Different interpolation types are available, see @ref InterpolationTypes.


@param[in] src The input Image.
@param[out] dst The output rotated Image.
@param[in] angle The rotation angle in degrees.
@param[in] scale Optional scaling factor.
@param[in] interp Interpolation type used. Default is InterpolationType::linear.

*/
void RotateFullImage2D(const ecvl::Image& src, ecvl::Image& dst, double angle, double scale = 1.0, InterpolationType interp = InterpolationType::linear);

} // namespace ecvl

#endif // ECVL_IMGPROC_H_