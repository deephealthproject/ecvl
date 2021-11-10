/*
* ECVL - European Computer Vision Library
* Version: 0.3.4
* copyright (c) 2021, Universit� degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include <ecvl/core/image.h>
#include "ecvl/core/fpga_hal.h"

#define ECVL_FPGA

namespace ecvl
{

FpgaHal* FpgaHal::GetInstance()
{
#ifndef ECVL_FPGA
    ECVL_ERROR_DEVICE_UNAVAILABLE(FPGA)
#endif // ECVL_FPGA

  printf("FPGA getinstance\n");

    static FpgaHal instance; 	// Guaranteed to be destroyed.
                               // Instantiated on first use.
    return &instance;
}

void FpgaHal::CopyImage(const Image& src, Image& dst)
{
    printf("FpgaHal::CopyImage not implemented\n"); exit(1);
}

void FpgaHal::ConvertTo(const Image& src, Image& dst, DataType dtype, bool saturate)
{
    printf("FpgaHal::ConvertTo not implemented\n"); exit(1);
}

void FpgaHal::RearrangeChannels(const Image& src, Image& dst, const std::vector<int>& bindings)
{
    printf("FpgaHal::RearrangeChannels not implemented\n"); exit(1);
}

} // namespace ecvl
