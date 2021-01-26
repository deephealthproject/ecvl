/*
* ECVL - European Computer Vision Library
* Version: 0.3.1
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
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

#include <cuda_runtime.h>

#include "ecvl/core/hal.h"
#include "ecvl/core/cuda/common.h"


namespace ecvl
{
class GpuHal : public HardwareAbstractionLayer
{
public:
    uint8_t* MemAllocate(size_t nbytes) override;

    void MemDeallocate(uint8_t* data) override
    {
        checkCudaError(cudaFree(data));
    }

    uint8_t* MemCopy(uint8_t* dst, const uint8_t* src, size_t nbytes) override;

    static GpuHal* GetInstance();

    void FromCpu(Image& src) override;
    
    void ToCpu(Image& src) override;
};
} // namespace ecvl


#endif // ECVL_GPU_HAL_H_
