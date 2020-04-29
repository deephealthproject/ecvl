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

#include "ecvl/core/cuda/gpu_hal.h"

#include <cuda_runtime.h>

#include <ecvl/core/image.h>
#include <ecvl/core/cuda/common.h>
#include <ecvl/core/cpu_hal.h>

namespace ecvl
{

GpuHal* GpuHal::GetInstance()
{
    static GpuHal instance;     // Guaranteed to be destroyed.
                                // Instantiated on first use.
    return &instance;
}

uint8_t* GpuHal::MemAllocate(size_t nbytes)
{
    uint8_t* devPtr;
    checkCudaError(cudaMalloc(&devPtr, nbytes));
    return devPtr;
}

uint8_t* GpuHal::MemCopy(uint8_t* dst, const uint8_t* src, size_t nbytes)
{
    checkCudaError(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice));
    return dst;
}

void GpuHal::FromCpu(Image& src)
{
    if (!src.contiguous_) {
        // The copy constructor creates a new contiguous image
        src = Image(src);
    }

    uint8_t* devPtr;
    checkCudaError(cudaMalloc(&devPtr, src.datasize_));
    checkCudaError(cudaMemcpy(devPtr, src.data_, src.datasize_, cudaMemcpyHostToDevice));
    src.hal_->MemDeallocate(src.data_);
    src.data_ = devPtr;
    src.hal_ = GpuHal::GetInstance();
    src.dev_ = Device::GPU;
}

void GpuHal::ToCpu(Image& src)
{
    if (!src.contiguous_) {
        // The copy constructor creates a new contiguous image
        src = Image(src);
    }

    src.hal_ = CpuHal::GetInstance();
    src.dev_ = Device::CPU;
    uint8_t* hostData = src.hal_->MemAllocate(src.datasize_);
    checkCudaError(cudaMemcpy(hostData, src.data_, src.datasize_, cudaMemcpyDeviceToHost));
    checkCudaError(cudaFree(src.data_));
    src.data_ = hostData;
}

} // namespace ecvl
