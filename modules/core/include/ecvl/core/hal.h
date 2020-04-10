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

#include "datatype.h"

#include "standard_errors.h"

#define ECVL_ERROR_DEVICE_UNAVAILABLE(device) throw std::runtime_error(ECVL_ERROR_MSG #device " device unavailable");

namespace ecvl {

/** @brief Enum class representing the ECVL available devices

@anchor Device
*/
enum class Device { 
    NONE,  /**< Special Device for empty images without any associated device */
    CPU,   /**< CPU Device */
    GPU,   /**< GPU Device */
    FPGA,  /**< FPGA Device */
};         
           
class Image;

/** @brief Hardware Abstraction Layer (HAL) is an abstraction layer to interact with a hardware device at a 
    general level

    HAL is an interface that allows ECVL to interact with hardwares devices at a general or abstract level
    rather than at a detailed hardware level. It represents a proxy to the actual function implementations 
    that must be device specific. 

    Actual HALs must inherit from this base class. Most of the memory handling methods must be overwritten.
    This base class also provides some general methods that can be shared by different devices.

*/
class HardwareAbstractionLayer {
public:

    static HardwareAbstractionLayer* Factory(Device dev, bool shallow = false);

    virtual uint8_t* MemAllocate(size_t nbytes) = 0;
    virtual void MemDeallocate(uint8_t* data) = 0;
    virtual uint8_t* MemCopy(uint8_t* dst, const uint8_t* src, size_t nbytes) = 0;
    virtual uint8_t* MemAllocateAndCopy(size_t nbytes, const uint8_t* src) {
        return MemCopy(MemAllocate(nbytes), src, nbytes);
    }

    // We don't need a virtual destructor because HALs are created as static objects using a singleton pattern
    // virtual ~HardwareAbstractionLayer() {}

    /** @brief Specific function which allocates data for a partially initialized image object

        This function delegates the operation of creating image data to the specific HAL. The default
        version assumes a contiguous image, so the strides are exactly those expected from the dims_ vector.
        Specific HALs could change the memory layout by operating on the specific fields.
    */
    virtual void Create(Image& img);
    virtual void Copy(const Image& src, Image& dst);

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