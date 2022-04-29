/*
* ECVL - European Computer Vision Library
* Version: 1.0.3
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#ifndef ECVL_CUDA_COMMON_H_
#define ECVL_CUDA_COMMON_H_

#include <string>

#include <ecvl/core/standard_errors.h>

#include <cuda_runtime.h>

namespace ecvl
{

static inline void checkCudaError(cudaError_t err)
{
    if (cudaSuccess != err)
        throw std::runtime_error(std::string(ECVL_ERROR_MSG) + std::string(cudaGetErrorString(err)));
}


#ifndef ecvlCudaSafeCall
#define ecvlCudaSafeCall(expr)  ecvl::checkCudaError(expr)
#endif

}

#endif // ECVL_CUDA_COMMON_H_ 