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

#ifndef ECVL_HARDWAREABSTRACTIONLAYER_H_
#define ECVL_HARDWAREABSTRACTIONLAYER_H_

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ecvl {

enum class Device { NONE, CPU, GPU, FPGA };

class Image;

class HardwareAbstractionLayer {
public:
    static HardwareAbstractionLayer* Factory(Device dev);

    virtual uint8_t* MemAllocate(size_t nbytes) = 0;
    virtual void MemDeallocate(uint8_t* data) = 0;
    virtual uint8_t* MemCopy(uint8_t* dst, const uint8_t* src, size_t nbytes) = 0;
    virtual uint8_t* MemAllocateAndCopy(size_t nbytes, const uint8_t* src) {
        return MemCopy(MemAllocate(nbytes), src, nbytes);
    }
    virtual ~HardwareAbstractionLayer() {}

    /** @brief Specific function which allocates data for a partially initialized image object

        This function delegates the operation of creating image data to the specific HAL. The default
        version assumes a contiguous image, so the strides are exactly those expected from the dims_ vector.
        Specific HALs could change the memory layout by operating on the specific fields.
    */
    virtual void Create(Image& img);
    virtual void Copy(const Image& src, Image& dst);
};

class CpuHal : public HardwareAbstractionLayer {
public:
    uint8_t* MemAllocate(size_t nbytes) override {
        return new uint8_t[nbytes];
    }
    void MemDeallocate(uint8_t* data) override {
        delete[] data;
    }
    uint8_t* MemCopy(uint8_t* dst, const uint8_t* src, size_t nbytes) override {
        return reinterpret_cast<uint8_t*>(std::memcpy(dst, src, nbytes));
    }

    static CpuHal* GetInstance();
};

class ShallowCpuHal : public CpuHal {
public:
    uint8_t* MemAllocate(size_t nbytes) override {
        throw std::runtime_error("ShallowCpuHal cannot allocate memory");
    }
    void MemDeallocate(uint8_t* data) override {}

    void Copy(const Image& src, Image& dst) override;

    static ShallowCpuHal* GetInstance();
};

class FpgaHal : public HardwareAbstractionLayer
{
public:
    uint8_t* MemAllocate(size_t nbytes) override
    {
        // Implement FPGA memory allocation
    }
    void MemDeallocate(uint8_t* data) override
    {
        // Implement FPGA memory deallocation
    }
    uint8_t* MemCopy(uint8_t* dst, const uint8_t* src, size_t nbytes) override
    {
        // Implement FPGA memory copy
    }

    static FpgaHal* GetInstance();
};

};
#endif // ECVL_HARDWAREABSTRACTIONLAYER_H_