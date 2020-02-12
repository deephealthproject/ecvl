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

#ifndef ECVL_IMGPROC_H_
#define ECVL_IMGPROC_H_

#include "image.h"
#include "support_opencv.h"

namespace ecvl {
/** @brief Enum class representing the ECVL thresholding types.

    @anchor ThresholdingType
 */
enum class ThresholdingType {
    BINARY,     /**< \f[\texttt{dst} (x,y) =  \fork{\texttt{maxval}}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{0}{otherwise}\f] */
    BINARY_INV, /**< \f[\texttt{dst} (x,y) =  \fork{0}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{maxval}}{otherwise}\f] */
};

/** @brief Enum class representing the ECVL interpolation types.

    @anchor InterpolationType
 */
enum class InterpolationType {
    nearest,    /**< Nearest neighbor interpolation */
    linear,     /**< Bilinear interpolation */
    area,       /**< Resampling using pixel area relation.
                     It may be a preferred method for image decimation,
                     as it gives moire-free results. But when the image
                     is zoomed, it is similar to the nearest method. */
    cubic,      /**< Bicubic interpolation */
    lanczos4    /**< Lanczos interpolation over 8x8 neighborhood */
};

/** @brief Resizes an Image to the specified dimensions

The function resizes Image src and outputs the result in dst.

@param[in] src The input Image.
@param[out] dst The output resized Image.
@param[in] newdims std::vector<int> that specifies the new size of each dimension.
            The vector size must match the src Image dimensions, excluding the color channel.
@param[in] interp InterpolationType to be used. Default is InterpolationType::linear.

*/
void ResizeDim(const ecvl::Image& src, ecvl::Image& dst, const std::vector<int>& newdims, InterpolationType interp = InterpolationType::linear);

/** @brief Resizes an Image by scaling the dimensions to a given scale factor

The function resizes Image src and outputs the result in dst.

@param[in] src The input Image.
@param[out] dst The output rescaled Image.
@param[in] scales std::vector<double> that specifies the scale to apply to each dimension.
            The vector size must match the src Image dimensions, excluding the color channel.
@param[in] interp InterpolationType to be used. Default is InterpolationType::linear.

*/
void ResizeScale(const ecvl::Image& src, ecvl::Image& dst, const std::vector<double>& scales, InterpolationType interp = InterpolationType::linear);

/** @brief Flips an Image

The Flip2D procedure vertically flips an Image.

@param[in] src The input Image.
@param[out] dst The output flipped Image.

*/
void Flip2D(const ecvl::Image& src, ecvl::Image& dst);

/** @brief Mirrors an Image

The Mirror2D procedure horizontally flips an Image.

@param[in] src The input Image.
@param[out] dst The output mirrored Image.

*/
void Mirror2D(const ecvl::Image& src, ecvl::Image& dst);

/** @brief Rotates an Image

The Rotate2D procedure rotates an Image of a given angle (expressed in degrees) in a clockwise manner, with respect to a
given center. The value of unknown pixels in the output Image are set to 0. The output Image is guaranteed to have the same
dimensions as the input one. An optional scale parameter can be provided: this won't change the output Image size, but the
image is scaled during rotation. Different interpolation types are available, see @ref InterpolationType.

@param[in] src The input Image.
@param[out] dst The output rotated Image.
@param[in] angle The rotation angle in degrees.
@param[in] center A std::vector<double> representing the coordinates of the rotation center.
            If empty, the center of the image is used.
@param[in] scale Optional scaling factor.
@param[in] interp Interpolation type used. Default is InterpolationType::linear.

*/
void Rotate2D(const ecvl::Image& src, ecvl::Image& dst, double angle, const std::vector<double>& center = {}, double scale = 1.0, InterpolationType interp = InterpolationType::linear);

/** @brief Rotates an Image resizing the output accordingly.

The RotateFullImage2D procedure rotates an Image of a given angle (expressed in degrees) in a clockwise manner.
The value of unknown pixels in the output Image are set to 0. The output Image is guaranteed to contain all the pixels
of the rotated image. Thus, its dimensions can be different from those of the input.
An optional scale parameter can be provided. Different interpolation types are available, see @ref InterpolationType.

@param[in] src The input Image.
@param[out] dst The rotated output Image.
@param[in] angle The rotation angle in degrees.
@param[in] scale Optional scaling factor.
@param[in] interp Interpolation type used. Default is InterpolationType::linear.

*/
void RotateFullImage2D(const ecvl::Image& src, ecvl::Image& dst, double angle, double scale = 1.0, InterpolationType interp = InterpolationType::linear);

/** @brief Copies the source Image into destination Image changing the color space.

The ChangeColorSpace procedure converts the color space of the source Image into the specified color space.
New data are copied into destination Image. Source and destination can be contiguous or not and can also
be the same Image.

@param[in] src The input Image to convert in the new color space.
@param[out] dst The output Image in the "new_type" color space.
@param[in] new_type The new color space in which the src Image must be converted.

*/
void ChangeColorSpace(const Image& src, Image& dst, ColorType new_type);

/** @brief Applies a fixed threshold to an input Image.

The Threshold function applies a fixed thresholding to an input Image. The function is useful to get a binary
image out of a grayscale (ColorType::GRAY) Image or to remove noise filtering out pixels with too small or too
large values. Anyway, the function can be applied to any input Image.  The pixels up to "thresh" value will be
set to 0, the pixels above this value will be set to "maxvalue" if "thresh_type" is ThresholdingType::BINARY
(default). The opposite will happen if "thresh_type" is ThresholdingType::BINARY_INV.

@bug Input and output Images may have different color spaces.

@param[in] src Input Image on which to apply the threshold.
@param[out] dst The output thresholded Image.
@param[in] thresh Threshold value.
@param[in] maxval The maximum values in the thresholded Image.
@param[in] thresh_type Type of threshold to be applied, see @ref ThresholdingType. The default value is ThresholdingType::BINARY.

*/
void Threshold(const Image& src, Image& dst, double thresh, double maxval, ThresholdingType thresh_type = ThresholdingType::BINARY);

/** @brief Calculates the Otsu thresholding value.

The OtsuThreshold function calculates the Otsu threshold value over a given input Image. the Image must by ColorType::GRAY.

@param[in] src Input Image on which to calculate the Otsu threshold value.

@return Otsu threshold value.
*/
int OtsuThreshold(const Image& src);

/** @brief Convolves an Image with a kernel

@param[in] src Input Image.
@param[out] dst Output Image.
@param[in] ker Convolution kernel.
@param[in] type Destination ecvl::DataType. If DataType::none, the same of src is used.

*/
void Filter2D(const Image& src, Image& dst, const Image& ker, DataType type = DataType::none /* type of border */);

/** @brief Convolves an Image with a couple of 1-dimensional kernels

@param[in] src Input Image.
@param[out] dst Output Image.
@param[in] kerX Convolution kernel for the X-axis.
@param[in] kerY Convolution kernel for the Y-axis.
@param[in] type Destination ecvl::DataType. If DataType::none, the same of src is used.

*/
void SeparableFilter2D(const Image& src, Image& dst, const std::vector<double>& kerX, const std::vector<double>& kerY, DataType type = DataType::none);

/** @brief Blurs an Image using a Gaussian kernel.

@param[in] src Input Image.
@param[out] dst Output Image.
@param[in] sizeX Horizontal size of the kernel. Must be positive and odd.
@param[in] sizeY Vertical size of the kernel. Must be positive and odd.
@param[in] sigmaX Gaussian kernel standard deviation in X direction.
@param[in] sigmaY Gaussian kernel standard deviation in Y direction. If zero, sigmaX is used. If both are zero, they are calculated from sizeX and sizeY.

*/
void GaussianBlur(const Image& src, Image& dst, int sizeX, int sizeY, double sigmaX, double sigmaY = 0);
void GaussianBlur(const Image& src, Image& dst, double sigma);

/** @brief Adds Laplace distributed noise to an Image.

@param[in] src Input Image.
@param[out] dst Output Image.
@param[in] std_dev Standard deviation of the noise generating distribution. Suggested values are around 255 * 0.05 for uint8 Images.

*/
void AdditiveLaplaceNoise(const Image& src, Image& dst, double std_dev);

/** @brief Adds Poisson distributed noise to an Image.

@param[in] src Input Image.
@param[out] dst Output Image.
@param[in] lambda Lambda parameter of the Poisson distribution.

*/
void AdditivePoissonNoise(const Image& src, Image& dst, double lambda);

/** @brief Adjust contrast by scaling each pixel value X to 255 * ((X/255) ** gamma).

@param[in] src Input Image.
@param[out] dst Output Image.
@param[in] gamma Exponent for the contrast adjustment.
*/
void GammaContrast(const Image& src, Image& dst, double gamma);

/** @brief Sets rectangular areas within an Image to zero.

@param[in] src Input Image.
@param[out] dst Output Image.
@param[in] p Probability of any rectangle being set to zero.
@param[in] drop_size Size of rectangles in percentage of the input Image.
@param[in] per_channel Whether to use the same value for all channels of a pixel or not.

*/
void CoarseDropout(const Image& src, Image& dst, double p, double drop_size, bool per_channel);

/** @brief Calculate the integral image of the source Image.

@param[in] src Input Image. It must be with ColorType::GRAY, "xyc" and DataType::uint8.
@param[out] dst Output Image.
@param[in] dst_type DataType of the destination Image.
*/
void IntegralImage(const Image& src, Image& dst, DataType dst_type = DataType::float64);

/** @brief Calculate the Non-Maxima suppression of the source Image.

@param[in] src Input Image. It must be with ColorType::GRAY, "xyc" and DataType::int32.
@param[out] dst Output Image.
*/
void NonMaximaSuppression(const Image& src, Image& dst);

/** @brief Get the `n` maximum values that are in the source Image.

@param[in] src Input Image. It must be with ColorType::GRAY, "xyc" and DataType::int32.
@param[in] n How many values must be returned.
@return vector of Point2i containing the coordinates of the n max values of the Image.
*/
std::vector<ecvl::Point2i> GetMaxN(const Image& src, size_t n);

/** @brief Labels connected components in an binary Image

@param[in] src Input Image. It must be with channels "xyc", only one color channel and DataType::uint8.
@param[out] dst Output Image.
*/
void ConnectedComponentsLabeling(const Image& src, Image& dst);

/** @brief Finds contours in a binary image

@param[in] src Input Image. It must be with channels "xyc", only one color channel and DataType::uint8.
@param[out] dst Output contours.
*/
void FindContours(const Image& src, std::vector<std::vector<Point2i>>& contours);

/** @brief Stack a sequence of Images along a new depth dimension (images dimensions must match)

@param[in] src vector of input Images. It must be with channels "xyc".
@param[out] dst Output Image.
*/
void Stack(const std::vector<Image>& src, Image& dst);

/** @brief Horizontal concatenation of images (with the same number of rows)

@param[in] src vector of input Images. It must be with channels "xyc".
@param[out] dst Output Image.
*/
void HConcat(const std::vector<Image>& src, Image& dst);

/** @brief Vertical concatenation of images (with the same number of columns)

@param[in] src vector of input Images. It must be with channels "xyc".
@param[out] dst Output Image.
*/
void VConcat(const std::vector<Image>& src, Image& dst);

enum class MorphTypes {
    MORPH_ERODE   , /**< see #erode */
    MORPH_DILATE  , /**< see #dilate */
    MORPH_OPEN    , /**< an opening operation  \f[\texttt{dst} = \mathrm{open} ( \texttt{src} , \texttt{element} )= \mathrm{dilate} ( \mathrm{erode} ( \texttt{src} , \texttt{element} ))\f] */
    MORPH_CLOSE   , /**< a closing operation */
                    /**< \f[\texttt{dst} = \mathrm{close} ( \texttt{src} , \texttt{element} )= \mathrm{erode} ( \mathrm{dilate} ( \texttt{src} , \texttt{element} ))\f]*/
    MORPH_GRADIENT, /**< a morphological gradient */
                    /**< \f[\texttt{dst} = \mathrm{morph\_grad} ( \texttt{src} , \texttt{element} )= \mathrm{dilate} ( \texttt{src} , \texttt{element} )- \mathrm{erode} ( \texttt{src} , \texttt{element} )\f] */
    MORPH_TOPHAT  , /**< "top hat" */
                    /**< \f[\texttt{dst} = \mathrm{tophat} ( \texttt{src} , \texttt{element} )= \texttt{src} - \mathrm{open} ( \texttt{src} , \texttt{element} )\f] */
    MORPH_BLACKHAT, /**< "black hat" */
                    /**< \f[\texttt{dst} = \mathrm{blackhat} ( \texttt{src} , \texttt{element} )= \mathrm{close} ( \texttt{src} , \texttt{element} )- \texttt{src}\f] */
    MORPH_HITMISS   /**< "hit or miss" */
                    /**<   .- Only supported for CV_8UC1 binary images. A tutorial can be found in the documentation */
};

void Morphology(const Image& src, Image& dst, MorphTypes op, Image& kernel,
    Point2i anchor = { -1, -1 },
    int iterations = 1,
    int borderType = 1 /*BORDER_CONSTANT*/,
    const int& 	borderValue = 0/*morphologyDefaultBorderValue()*/
);
enum class InpaintTypes
{
    INPAINT_NS, //!< Use Navier-Stokes based method
    INPAINT_TELEA //!< Use the algorithm proposed by Alexandru Telea @cite Telea04
};

void Inpaint(const Image& src, Image& dst, const Image& inpaintMask, double inpaintRadius, InpaintTypes flag = InpaintTypes::INPAINT_TELEA);

void MeanStdDev(const Image& src, std::vector<double>& mean, std::vector<double>& stddev);

/** @example example_imgproc.cpp
 Imgproc example.
*/
} // namespace ecvl

#endif // ECVL_IMGPROC_H_