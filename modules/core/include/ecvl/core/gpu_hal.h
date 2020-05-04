/*
* ECVL - European Computer Vision Library
* Version: 0.1
* copyright (c) 2020, Universit� degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#ifndef ECVL_GPU_HAL_H_
#define ECVL_GPU_HAL_H_

#include "ecvl/core/hal.h"

namespace ecvl
{
class GpuHal : public HardwareAbstractionLayer
{
public:
    uint8_t* MemAllocate(size_t nbytes) override
    {
        ECVL_ERROR_NOT_IMPLEMENTED_WHAT("GpuHal::MemAllocate")
    }
    void MemDeallocate(uint8_t* data) override
    {
        ECVL_ERROR_NOT_IMPLEMENTED_WHAT("GpuHal::MemDeallocate")
    }
    uint8_t* MemCopy(uint8_t* dst, const uint8_t* src, size_t nbytes) override
    {
        ECVL_ERROR_NOT_IMPLEMENTED_WHAT("GpuHal::MemCopy")
    }

    static GpuHal* GetInstance();

    void FromCpu(Image& src) override
    {
        ECVL_ERROR_NOT_IMPLEMENTED_WHAT("GpuHal::FromCpu")
    }
    void ToCpu(Image& src) override
    {
        ECVL_ERROR_NOT_IMPLEMENTED_WHAT("GpuHal::ToCpu")
    }
};
} // namespace ecvl
#endif // ECVL_GPU_HAL_H_