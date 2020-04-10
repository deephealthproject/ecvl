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

#include "ecvl/core/cpu_hal.h"

#include "ecvl/core/image.h"
#include "ecvl/core/saturate_cast.h"

namespace ecvl
{

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



} // namespace ecvl





