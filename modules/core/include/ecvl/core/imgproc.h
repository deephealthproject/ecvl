/*
* ECVL - European Computer Vision Library
* Version: 0.2.1
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

#include <random>
#include "image.h"
#include "saturate_cast.h"

namespace ecvl
{
/** @brief Enum class representing the ECVL thresholding types.

    @anchor ThresholdingType
 */
enum class ThresholdingType
{
    BINARY,     /**< \f[\texttt{dst} (x,y) =  \fork{\texttt{maxval}}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{0}{otherwise}\f] */
    BINARY_INV, /**< \f[\texttt{dst} (x,y) =  \fork{0}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{maxval}}{otherwise}\f] */
};

/** @brief Enum class representing the ECVL interpolation types.

    @anchor InterpolationType
 */
enum class InterpolationType
{
    nearest,    /**< Nearest neighbor interpolation */
    linear,     /**< Bilinear interpolation */
    area,       /**< Resampling using pixel area relation.
                     It may be a preferred method for image decimation,
                     as it gives moire-free results. But when the image
                     is zoomed, it is similar to the nearest method. */
    cubic,      /**< Bicubic interpolation */
    lanczos4    /**< Lanczos interpolation over 8x8 neighborhood */
};

/** @brief Given an InterpolationType, the GetOpenCVInterpolation function returns the associated OpenCV enum value.

@param[in] interp Interpolation type, see @ref InterpolationType.

@return Associated OpenCV enum value.
*/
int GetOpenCVInterpolation(InterpolationType interp);

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
large values. Anyway, the function can be applied to any input Image. The pixels up to "thresh" value will be
set to 0, the pixels above this value will be set to "maxvalue" if "thresh_type" is ThresholdingType::BINARY
(default). The opposite will happen if "thresh_type" is ThresholdingType::BINARY_INV.

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
@param[out] contours Output contours.
*/
void FindContours(const Image& src, std::vector<std::vector<Point2i>>& contours);

/** @brief Stack a sequence of Images along the depth dimension (images width and height must match)

@param[in] src vector of input Images. It must be with channels "xyc".
@param[out] dst Output Image.
*/
void Stack(const std::vector<Image>& src, Image& dst);

/** @brief Horizontal concatenation of images (with the same number of rows)

@param[in] src vector of input Images.
@param[out] dst Output Image.
*/
void HConcat(const std::vector<Image>& src, Image& dst);

/** @brief Vertical concatenation of images (with the same number of columns)

@param[in] src vector of input Images.
@param[out] dst Output Image.
*/
void VConcat(const std::vector<Image>& src, Image& dst);

/** @brief Enum class representing the ECVL morphology types.

    @anchor MorphType
 */
enum class MorphType
{
    MORPH_ERODE   ,
    MORPH_DILATE  ,
    MORPH_OPEN    , /**< an opening operation  \f[\texttt{dst} = \mathrm{open} ( \texttt{src} , \texttt{element} )= \mathrm{dilate} ( \mathrm{erode} ( \texttt{src} , \texttt{element} ))\f] */
    MORPH_CLOSE   , /**< a closing operation */
                    /**< \f[\texttt{dst} = \mathrm{close} ( \texttt{src} , \texttt{element} )= \mathrm{erode} ( \mathrm{dilate} ( \texttt{src} , \texttt{element} ))\f]*/
    MORPH_GRADIENT, /**< a morphological gradient */
                    /**< \f[\texttt{dst} = \mathrm{morph\_grad} ( \texttt{src} , \texttt{element} )= \mathrm{dilate} ( \texttt{src} , \texttt{element} )- \mathrm{erode} ( \texttt{src} , \texttt{element} )\f] */
    MORPH_TOPHAT  , /**< "top hat" */
                    /**< \f[\texttt{dst} = \mathrm{tophat} ( \texttt{src} , \texttt{element} )= \texttt{src} - \mathrm{open} ( \texttt{src} , \texttt{element} )\f] */
    MORPH_BLACKHAT, /**< "black hat" */
                    /**< \f[\texttt{dst} = \mathrm{blackhat} ( \texttt{src} , \texttt{element} )= \mathrm{close} ( \texttt{src} , \texttt{element} )- \texttt{src}\f] */
    MORPH_HITMISS   /**< "hit or miss". */
                    /**<   Only supported for DataType::uint8 binary images.*/
};

/** @brief Enum class representing the ECVL border types.

    @anchor BorderType
 */
enum class BorderType
{
    BORDER_CONSTANT,     //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
    BORDER_REPLICATE,    //!< `aaaaaa|abcdefgh|hhhhhhh`
    BORDER_REFLECT,      //!< `fedcba|abcdefgh|hgfedcb`
    BORDER_WRAP,         //!< `cdefgh|abcdefgh|abcdefg`
    BORDER_REFLECT_101,  //!< `gfedcb|abcdefgh|gfedcba`
    BORDER_TRANSPARENT   //!< `uvwxyz|abcdefgh|ijklmno`
};

/** @brief Performs advanced morphological transformations using an erosion and dilation as basic operations.

@param[in] src Input Image.
@param[out] dst Output Image.
@param[in] op Type of a morphological operation, see MorphType.
@param[in] kernel Structuring element.
@param[in] anchor Anchor position with the kernel. Negative values mean that the anchor is at the kernel center.
@param[in] iterations Number of times erosion and dilation are applied.
@param[in] border_type Pixel extrapolation method, see BorderType. BorderType::BORDER_WRAP is not supported.
@param[in] border_value Border value in case of a constant border.
*/
void Morphology(const Image& src, Image& dst, MorphType op, Image& kernel,
    Point2i anchor = { -1, -1 },
    int iterations = 1,
    BorderType border_type = BorderType::BORDER_CONSTANT,
    const int& border_value = 0 /*morphologyDefaultBorderValue()*/
);

/** @brief Enum class representing the ECVL inpaint types.

    @anchor InpaintType
 */
enum class InpaintType
{
    INPAINT_NS,     //!< Use Navier-Stokes based method
    INPAINT_TELEA   //!< Use the algorithm proposed by Alexandru Telea
};

/** @brief Restores the selected region in an image using the region neighborhood.

@param[in] src Input Image.
@param[out] dst Output Image.
@param[in] inpaintMask Inpainting mask, an Image with 1-channel and DataType::uint8. Non-zero pixels indicate the area that needs to be inpainted.
@param[in] inpaintRadius Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
@param[in] flag Inpainting method that could be InpaintType::INPAINT_NS or InpaintType::INPAINT_TELEA.
*/
void Inpaint(const Image& src, Image& dst, const Image& inpaintMask, double inpaintRadius, InpaintType flag = InpaintType::INPAINT_TELEA);

/** @brief Calculates the mean and the standard deviation of an Image.

@param[in] src Input Image.
@param[out] mean Mean of the Image pixels.
@param[out] stddev Standard deviation of the Image pixels.
*/
void MeanStdDev(const Image& src, std::vector<double>& mean, std::vector<double>& stddev);

/** @brief Swap rows and columns of an Image.

@param[in] src Input Image.
@param[out] dst Output transposed Image.
*/
void Transpose(const Image& src, Image& dst);

/** @brief Randomly stretch or reduce each cell of the grid in which the input Image is divided into.
Based on https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py#L1175

@param[in] src Input Image.
@param[out] dst Output Image.
@param[in] num_steps Count of grid cells on each side.
@param[in] distort_limit Range of distortion steps.
@param[in] interp Interpolation type used. Default is InterpolationType::linear.
@param[in] border_type Flag used to specify the pixel extrapolation method. Default is BorderType::BORDER_REFLECT_101.
@param[in] border_value Padding value if border_type is BorderType::BORDER_CONSTANT.
@param[in] seed Seed to use for this function's random number generator.
*/
void GridDistortion(const Image& src, Image& dst,
    int num_steps = 5,
    const std::array<float, 2>& distort_limit = { -0.3f, 0.3f },
    InterpolationType interp = InterpolationType::linear,
    BorderType border_type = BorderType::BORDER_REFLECT_101,
    const int& border_value = 0,
    const unsigned seed = std::default_random_engine::default_seed
);

/** @brief Elastic deformation of input Image.
Based on https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py#L1235.

@param[in] src Input Image.
@param[out] dst Output Image.
@param[in] alpha Scaling factor that controls the intensity of the deformation.
@param[in] sigma Gaussian kernel standard deviation
@param[in] interp Interpolation type used. If src is DataType::int8 or DataType::int32, Interpolation::nearest is used. Default is InterpolationType::linear.
@param[in] border_type Flag used to specify the pixel extrapolation method. Default is BorderType::BORDER_REFLECT_101.
@param[in] border_value Padding value if border_type is BorderType::BORDER_CONSTANT.
@param[in] seed Seed to use for this function's random number generator.
*/
void ElasticTransform(const Image& src, Image& dst, 
    double alpha = 34.,
    double sigma = 4.,
    InterpolationType interp = InterpolationType::linear,
    BorderType border_type = BorderType::BORDER_REFLECT_101,
    const int& border_value = 0,
    const unsigned seed = std::default_random_engine::default_seed
);

/** @brief Barrel / pincushion distortion.
Based on https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py#L1114

@param[in] src Input Image.
@param[out] dst Output Image.
@param[in] distort_limit Range to randomly select the intensity of the distortion.
@param[in] shift_limit Range of image shifting.
@param[in] interp Interpolation type used. Default is InterpolationType::linear.
@param[in] border_type Flag used to specify the pixel extrapolation method. Default is BorderType::BORDER_REFLECT_101.
@param[in] border_value Padding value if border_type is BorderType::BORDER_CONSTANT.
@param[in] seed Seed to use for this function's random number generator.
*/
void OpticalDistortion(const Image& src, Image& dst,
    const std::array<float, 2>& distort_limit = { -0.3f, 0.3f },
    const std::array<float, 2>& shift_limit = { -0.1f, 0.1f },
    InterpolationType interp = InterpolationType::linear,
    BorderType border_type = BorderType::BORDER_REFLECT_101,
    const int& border_value = 0,
    const unsigned seed = std::default_random_engine::default_seed
);

/** @brief Adds salt noise (white pixels) to an Image.

@param[in] src Input Image.
@param[out] dst Output Image.
@param[in] p Probability of replacing a pixel with salt noise.
@param[in] per_channel If true, noise is not considered pixel-wise but channel-wise.
@param[in] seed Seed to use for this function's random number generator.
*/
void Salt(const Image& src, Image& dst, double p, bool per_channel = false, const unsigned seed = std::default_random_engine::default_seed);

/** @brief Adds pepper noise (black pixels) to an Image.

@param[in] src Input Image.
@param[out] dst Output Image.
@param[in] p Probability of replacing a pixel with pepper noise.
@param[in] per_channel If true, noise is not considered pixel-wise but channel-wise.
@param[in] seed Seed to use for this function's random number generator.
*/
void Pepper(const Image& src, Image& dst, double p, bool per_channel = false, const unsigned seed = std::default_random_engine::default_seed);

/** @brief Adds salt and pepper noise (white and black pixels) to an Image. White and black pixels are equally likely.

@param[in] src Input Image.
@param[out] dst Output Image.
@param[in] p Probability of replacing a pixel with salt and pepper noise.
@param[in] per_channel If true, noise is not considered pixel-wise but channel-wise.
@param[in] seed Seed to use for this function's random number generator.
*/
void SaltAndPepper(const Image& src, Image& dst, double p, bool per_channel = false, const unsigned seed = std::default_random_engine::default_seed);

/** @brief Corrects each voxel's time-series.
    Slice timing correction works by using (Hanning-windowed) sinc interpolation to shift each time-series by an appropriate fraction of a
    TR relative to the middle of the TR period. The default slice order acquisition is from the bottom of the brain to the top.

@param[in] src Input Image. It must be with channels "xyzt" and with spacings (distance between consecutive voxels on each dimensions).
@param[out] dst Output Image. It will be with DataType::float32.
@param[in] odd Slices were acquired with interleaved order (0, 2, 4... 1, 3, 5...)
@param[in] down Slices were acquired from the top of the brain to the bottom
*/
void SliceTimingCorrection(const Image& src, Image& dst, bool odd = false, bool down = false);

/** @brief Calculate all raw image moments of the source Image up to the specified order.

When working with a 2D Image, naming the pixel intensities as \f$I(x,y)\f$, raw image moments \f$M_{ij}\f$ are calculated with the following formula:

\f$
M_{ij} = \sum_x{\sum_y{x^iy^jI(x,y)}}
\f$

The following properties can be derived from raw image moments:
- Area (for binary images) or sum of grey level (for grayscale images): \f$M_{00} \f$, accessible through moments(0,0) <em>i.e.</em> moments(x,y);
- Centroid: \f$\{\bar{x}, \bar{y}\} = \{\frac{M_{10}}{M_{00}}, \frac{M_{01}}{M_{00}}\}\f$.

The formula above can be accordingly extended when working with higher dimensions. <b>Note that raw moments are neither translation, scale nor rotation invariant</b>.
Moments are stored in the output Image in the same order as for source channels.

@param[in] src Input Image on which calculating row moments up to the specified order. It must be a grayscale (ColorType::GRAY) or a data (ColorType::none) Image.
@param[out] moments Output data (ColorType:none) Image containing the computed raw image moments. The moments DataType is specified by the type parameter. The size of the Image will be (order + 1, order + 1)
@param[in] order Raw image moments will be calculated up to the specified order. Default is 3.
@param[in] type Specify the ecvl::DataType to be used for the moments Image. Default is DataType::float64.
*/
void Moments(const Image& src, Image& moments, int order = 3, DataType type = DataType::float64);

/** @example example_imgproc.cpp
 Imgproc example.
*/
} // namespace ecvl

#endif // ECVL_IMGPROC_H_