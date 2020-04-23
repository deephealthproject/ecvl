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

#include <ecvl/core/cuda/common.h>

namespace ecvl
{

GpuHal* GpuHal::GetInstance()
{

    static GpuHal instance; 	// Guaranteed to be destroyed.
                                // Instantiated on first use.
    return &instance;
}

} // namespace ecvl
