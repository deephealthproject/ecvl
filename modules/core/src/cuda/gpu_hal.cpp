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
    if (src.hal_ != CpuHal::GetInstance()) {
        throw std::runtime_error(ECVL_ERROR_MSG "");
    }

    if (!src.contiguous_) {
        // Make src contiguous
        size_t new_datasize = std::accumulate(std::begin(src.dims_), std::end(src.dims_), size_t(src.elemsize_), std::multiplies<size_t>());
        uint8_t* new_data = new uint8_t[new_datasize];
        int dims = vsize(src.dims_);

        int snd = 0;    // smallest noncontiguous dim
        int run_size = src.elemsize_;
        for (int i = 0; i < dims; i++) {
            if (run_size != src.strides_[i]) {
                break;
            }
            snd++;
            run_size = run_size * src.dims_[i];
        }

        if (snd >= dims) {
            ECVL_ERROR_NOT_REACHABLE_CODE
        }

        std::vector<int> pos(dims, 0);
        uint8_t* new_data_ptr = new_data;
        uint8_t* data_ptr = src.data_;
        while (true) {

            std::memcpy(new_data_ptr, data_ptr, run_size);
            new_data_ptr += run_size;

            // Increment source pointer
            int dim;
            for (dim = snd; dim < dims; ++dim) {
                ++pos[dim];
                data_ptr += src.strides_[dim];
                if (pos[dim] != src.dims_[dim])
                    break;
                // Back to dimension starting position
                pos[dim] = 0;
                data_ptr -= src.dims_[dim] * src.strides_[dim];
            }
            if (dim == dims)
                break;
        }

        delete[] src.data_;
        src.data_ = new_data;
        src.datasize_ = new_datasize;
        src.strides_[0] = src.elemsize_;
        for (int i = 1; i < dims; ++i) {
            src.strides_[i] = src.strides_[i - 1] * src.dims_[i - 1];
        }
        src.contiguous_ = true;
    }

    uint8_t* devPtr;
    checkCudaError(cudaMalloc(&devPtr, src.datasize_));
    checkCudaError(cudaMemcpy(devPtr, src.data_, src.datasize_, cudaMemcpyHostToDevice));
    delete[] src.data_;
    src.data_ = devPtr;
    src.hal_ = GpuHal::GetInstance();
    src.dev_ = Device::GPU;
}

void GpuHal::ToCpu(Image& src)
{
    if (!src.contiguous_) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    uint8_t* hostData = new uint8_t[src.datasize_];
    checkCudaError(cudaMemcpy(hostData, src.data_, src.datasize_, cudaMemcpyDeviceToHost));
    checkCudaError(cudaFree(src.data_));
    src.data_ = hostData;
    src.hal_ = CpuHal::GetInstance();
    src.dev_ = Device::CPU;
}


} // namespace ecvl
