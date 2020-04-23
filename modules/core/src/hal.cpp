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

#include "ecvl/core/hal.h"

#include <cassert>

#include "ecvl/core/cpu_hal.h"
#include "ecvl/core/fpga_hal.h"

#if defined ECVL_GPU
#include "ecvl/core/cuda/gpu_hal.h"
#endif // ECVL_GPU

#include "ecvl/core/image.h"

namespace ecvl
{

HardwareAbstractionLayer* HardwareAbstractionLayer::Factory(Device dev, bool shallow)
{
    switch (dev) {
    case ecvl::Device::NONE:
        throw std::runtime_error("This is a big problem. You should never try to obtain a HAL from NONE device.");
    case ecvl::Device::CPU:
        if (shallow) {
            return ShallowCpuHal::GetInstance();
        }
        else {
            return CpuHal::GetInstance();
        }
    case ecvl::Device::GPU:
#if defined ECVL_GPU
        return GpuHal::GetInstance();
#else
        ECVL_ERROR_DEVICE_UNAVAILABLE(GPU)
#endif
    case ecvl::Device::FPGA:
        return FpgaHal::GetInstance();
    default:
        ECVL_ERROR_NOT_REACHABLE_CODE
    }
};

void HardwareAbstractionLayer::Create(Image& img)
{
    img.SetDefaultStrides();
    img.SetDefaultDatasize();
    img.data_ = MemAllocate(img.datasize_);
}

void HardwareAbstractionLayer::Copy(const Image& src, Image& dst)
{
    assert(src.dev_ == dst.dev_);
    if (src.contiguous_) {
        dst.data_ = dst.hal_->MemAllocateAndCopy(src.datasize_, src.data_);
    }
    else {
        // When copying a non contiguous image, we make it contiguous (is this choice ok?)
        dst.contiguous_ = true;
        dst.hal_->Create(dst);
        // Copy with iterators: this is SUPER SLOW!
        // TODO: optimize so that we can memcpy one block at a time on the first dimension
        // This will require Iterators to increment more than one
        auto p = dst.data_;
        auto i = src.Begin<uint8_t>(), e = src.End<uint8_t>();
        for (; i != e; ++i) {
            dst.hal_->MemCopy(p++, i.ptr_, dst.elemsize_);
        }
    }
}

} // namespace ecvl