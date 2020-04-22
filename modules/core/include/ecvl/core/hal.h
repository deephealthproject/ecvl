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

#ifndef ECVL_HAL_H_
#define ECVL_HAL_H_

#include <cstdint>
#include <vector>

#include "datatype.h"

#include "standard_errors.h"

namespace ecvl
{
/** @brief Enum class representing the ECVL available devices

@anchor Device
*/
enum class Device
{
    NONE,  /**< Special Device for empty images without any associated device */
    CPU,   /**< CPU Device */
    GPU,   /**< GPU Device */
    FPGA,  /**< FPGA Device */
};

enum class ThresholdingType;
enum class InterpolationType;
enum class ColorType;
enum class MorphTypes;
enum class InpaintTypes;

class Image;

/** @brief Hardware Abstraction Layer (HAL) is an abstraction layer to interact with a hardware device at a
    general level

    HAL is an interface that allows ECVL to interact with hardwares devices at a general or abstract level
    rather than at a detailed hardware level. It represents a proxy to the actual function implementations
    that must be device specific.

    Actual HALs must inherit from this base class. Most of the memory handling methods must be overwritten.
    This base class also provides some general methods that can be shared by different devices.

*/
class HardwareAbstractionLayer
{
public:

    static HardwareAbstractionLayer* Factory(Device dev, bool shallow = false);

    virtual uint8_t* MemAllocate(size_t nbytes) = 0;
    virtual void MemDeallocate(uint8_t* data) = 0;
    virtual uint8_t* MemCopy(uint8_t* dst, const uint8_t* src, size_t nbytes) = 0;
    virtual uint8_t* MemAllocateAndCopy(size_t nbytes, const uint8_t* src)
    {
        return MemCopy(MemAllocate(nbytes), src, nbytes);
    }

    // We don't need a virtual destructor because HALs are created as static objects using a singleton pattern
    // virtual ~HardwareAbstractionLayer() {}

    virtual void FromCpu(Image& src) { ECVL_ERROR_NOT_IMPLEMENTED };
    virtual void ToCpu(Image& src) { ECVL_ERROR_NOT_IMPLEMENTED };

    /** @brief Specific function which allocates data for a partially initialized image object

        This function delegates the operation of creating image data to the specific HAL. The default
        version assumes a contiguous image, so the strides are exactly those expected from the dims_ vector.
        Specific HALs could change the memory layout by operating on the specific fields.
    */
    virtual void Create(Image& img);

    virtual void Copy(const Image& src, Image& dst);

    /** @brief Function for copying data from image of one type to one of another type

        Probably this could be merged with Copy. The idea is to have a function which allows for changing
        the datatype. Nevertheless, dst data has already been correctly initialized.
    */
    virtual void CopyImage(const Image& src, Image& dst) { ECVL_ERROR_NOT_IMPLEMENTED }

    /** @brief Changes the order of the Image dimensions.

    The RearrangeChannels procedure changes the order of the input Image dimensions saving
    the result into the output Image. The new order of dimensions is specified as a vector of
    ints, telling where each dimension should be in the destination.

    @param[in] src Input Image on which to rearrange dimensions.
    @param[out] dst The output rearranged Image. Can be the src Image.
    @param[in] bindings Desired order of Image channels.
    */
    virtual void RearrangeChannels(const Image& src, Image& dst, const std::vector<int>& bindings) { ECVL_ERROR_NOT_IMPLEMENTED }

    virtual void ResizeDim(const ecvl::Image& src, ecvl::Image& dst, const std::vector<int>& newdims, InterpolationType interp) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void ResizeScale(const Image& src, Image& dst, const std::vector<double>& scales, InterpolationType interp) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void Flip2D(const ecvl::Image& src, ecvl::Image& dst) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void Mirror2D(const ecvl::Image& src, ecvl::Image& dst) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void Rotate2D(const ecvl::Image& src, ecvl::Image& dst, double angle, const std::vector<double>& center, double scale, InterpolationType interp) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void RotateFullImage2D(const ecvl::Image& src, ecvl::Image& dst, double angle, double scale, InterpolationType interp) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void ChangeColorSpace(const Image& src, Image& dst, ColorType new_type) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void Threshold(const Image& src, Image& dst, double thresh, double maxval, ThresholdingType thresh_type) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual std::vector<double> Histogram(const Image& src) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual int OtsuThreshold(const Image& src) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void Filter2D(const Image& src, Image& dst, const Image& ker, DataType type) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void SeparableFilter2D(const Image& src, Image& dst, const std::vector<double>& kerX, const std::vector<double>& kerY, DataType type) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void GaussianBlur(const Image& src, Image& dst, int sizeX, int sizeY, double sigmaX, double sigmaY) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void AdditiveLaplaceNoise(const Image& src, Image& dst, double std_dev) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void AdditivePoissonNoise(const Image& src, Image& dst, double lambda) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void GammaContrast(const Image& src, Image& dst, double gamma) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void CoarseDropout(const Image& src, Image& dst, double p, double drop_size, bool per_channel) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void IntegralImage(const Image& src, Image& dst, DataType dst_type) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void NonMaximaSuppression(const Image& src, Image& dst) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual std::vector<ecvl::Point2i> GetMaxN(const Image& src, size_t n) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void ConnectedComponentsLabeling(const Image& src, Image& dst) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void FindContours(const Image& src, std::vector<std::vector<ecvl::Point2i>>& contours) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void Stack(const std::vector<Image>& src, Image& dst) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void HConcat(const std::vector<Image>& src, Image& dst) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void VConcat(const std::vector<Image>& src, Image& dst) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void Morphology(const Image& src, Image& dst, MorphTypes op, Image& kernel, Point2i anchor, int iterations, int borderType, const int& borderValue) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void Inpaint(const Image& src, Image& dst, const Image& inpaintMask, double inpaintRadius, InpaintTypes flag) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void MeanStdDev(const Image& src, std::vector<double>& mean, std::vector<double>& stddev) { ECVL_ERROR_NOT_IMPLEMENTED }

    virtual bool IsOwner() const { return true; };

    virtual void Neg(const Image& src, Image& dst, DataType dst_type, bool saturate) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void Add(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void Sub(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void Mul(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { ECVL_ERROR_NOT_IMPLEMENTED }
    virtual void Div(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { ECVL_ERROR_NOT_IMPLEMENTED }

#define ECVL_TUPLE(name, size, type, ...) \
    virtual void Add(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) { ECVL_ERROR_NOT_IMPLEMENTED } \
    virtual void Add(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { ECVL_ERROR_NOT_IMPLEMENTED } \
                                                                                                                               \
    virtual void Sub(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) { ECVL_ERROR_NOT_IMPLEMENTED } \
    virtual void Sub(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { ECVL_ERROR_NOT_IMPLEMENTED } \
                                                                                                                               \
    virtual void Mul(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) { ECVL_ERROR_NOT_IMPLEMENTED } \
    virtual void Mul(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { ECVL_ERROR_NOT_IMPLEMENTED } \
                                                                                                                               \
    virtual void Div(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) { ECVL_ERROR_NOT_IMPLEMENTED } \
    virtual void Div(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { ECVL_ERROR_NOT_IMPLEMENTED } \

#include "datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE
};
} // namespace ecvl
#endif // ECVL_HAL_H_