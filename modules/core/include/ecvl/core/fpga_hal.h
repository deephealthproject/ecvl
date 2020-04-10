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

#ifndef ECVL_FPGA_HAL_H_
#define ECVL_FPGA_HAL_H_

#include "ecvl/core/hal.h"

namespace ecvl
{

class FpgaHal : public HardwareAbstractionLayer
{
public:
    uint8_t* MemAllocate(size_t nbytes) override
    {
        ECVL_ERROR_NOT_IMPLEMENTED_WHAT("FpgaHal::MemAllocate")
    }
    void MemDeallocate(uint8_t* data) override
    {
        ECVL_ERROR_NOT_IMPLEMENTED_WHAT("FpgaHal::MemDeallocate")
    }
    uint8_t* MemCopy(uint8_t* dst, const uint8_t* src, size_t nbytes) override
    {
        ECVL_ERROR_NOT_IMPLEMENTED_WHAT("FpgaHal::MemCopy")
    }

    static FpgaHal* GetInstance();
};

} // namespace ecvl
#endif // ECVL_FPGA_HAL_H_
