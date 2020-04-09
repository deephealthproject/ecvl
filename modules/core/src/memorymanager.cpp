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

#include "ecvl/core/memorymanager.h"
#include "ecvl/core/image.h"

namespace ecvl {

HardwareAbstractionLayer* HardwareAbstractionLayer::Factory(Device dev)
{
    switch (dev)
    {
    case ecvl::Device::NONE:
        throw std::runtime_error("This is a big problem. You should never try to obtain a HAL from NONE device.");
    case ecvl::Device::CPU:
        return CpuHal::GetInstance();
    case ecvl::Device::GPU:
        throw std::runtime_error("GpuHal not implemented");
    case ecvl::Device::FPGA:
        throw std::runtime_error("FpgaHal not implemented");
    default:
        throw std::runtime_error("This is not the error you're looking for.");
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

CpuHal* CpuHal::GetInstance()
{
    static CpuHal instance;	// Guaranteed to be destroyed.
                            // Instantiated on first use.
    return &instance;
}

void ShallowCpuHal::Copy(const Image& src, Image& dst)
{
    // Copying from shallow -> destination becomes owner of the new data
    dst.hal_ = CpuHal::GetInstance();
    dst.hal_->Copy(src, dst);
}

ShallowCpuHal* ShallowCpuHal::GetInstance()
{
    static ShallowCpuHal instance;	// Guaranteed to be destroyed.
                                            // Instantiated on first use.
    return &instance;
}

};