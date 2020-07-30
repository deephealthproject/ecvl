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

#include "ecvl/core/hal.h"

#include <cassert>

#include "ecvl/core/cpu_hal.h"


#if defined ECVL_WITH_FPGA
#include "ecvl/core/fpga_hal.h"
#endif // ECVL_WITH_FPGA

#if defined ECVL_GPU
#include "ecvl/core/cuda/gpu_hal.h"
#endif // ECVL_GPU

#include "ecvl/core/image.h"
#include <iostream>

namespace ecvl
{
using namespace std;
HardwareAbstractionLayer* HardwareAbstractionLayer::Factory(Device dev, bool shallow)
{
	
    switch (dev) {
    case ecvl::Device::NONE:
        throw std::runtime_error("This is a big problem. You should never try to obtain a HAL from NONE device.");
    case ecvl::Device::CPU:
        if (shallow) {
			cout << "retorna shallow hal" << endl;
            return ShallowCpuHal::GetInstance();
        }
        else {
			cout << "retorna cpu hal" << endl;
            return CpuHal::GetInstance();
        }
    case ecvl::Device::GPU:
#if defined ECVL_GPU
        return GpuHal::GetInstance();
#else
        ECVL_ERROR_DEVICE_UNAVAILABLE(GPU)
#endif
    case ecvl::Device::FPGA:
#if defined ECVL_WITH_FPGA
		cout << "retorna fpga hal" << endl;
        return FpgaHal::GetInstance();
#else
	ECVL_ERROR_DEVICE_UNAVAILABLE(FPGA)
#endif
    default:
        ECVL_ERROR_NOT_REACHABLE_CODE
    }
};

void HardwareAbstractionLayer::Create(Image& img)
{

    img.SetDefaultStrides();
    img.SetDefaultDatasize();
    img.data_ = MemAllocate(img.datasize_);
	cout << "create image hal.cpp END" << endl;
}

void HardwareAbstractionLayer::Copy(const Image& src, Image& dst)
{
	cout << "entra a copiar imagen" << endl;
    assert(src.dev_ == dst.dev_);
    if (src.contiguous_) {
        dst.data_ = dst.hal_->MemAllocateAndCopy(src.datasize_, src.data_);
    }
    else {
        // When copying a non contiguous image, we make it contiguous (is this choice ok?)
        dst.contiguous_ = true;
        dst.hal_->Create(dst);

        int ndims = vsize(src.dims_);
        int sncd;    // smallest non-contiguous dimension
        for (sncd = 0; sncd < ndims && src.strides_[sncd] == dst.strides_[sncd]; ++sncd);

        std::vector<int> pos(ndims, 0);
        uint8_t* dst_data_ptr = dst.data_;
        const uint8_t* src_data_ptr = src.data_;
        while (true) {
            dst.hal_->MemCopy(dst_data_ptr, src_data_ptr, dst.strides_[sncd]);
            dst_data_ptr += dst.strides_[sncd];

            // Increment source pointer
            int dim;
            for (dim = sncd; dim < ndims; ++dim) {
                ++pos[dim];
                src_data_ptr += src.strides_[dim];
                if (pos[dim] != src.dims_[dim])
                    break;
                // Back to dimension starting position
                pos[dim] = 0;
                src_data_ptr -= src.dims_[dim] * src.strides_[dim];
            }
            if (dim == ndims)
                break;
        }
    }
}

} // namespace ecvl