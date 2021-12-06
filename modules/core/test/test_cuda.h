/*
* ECVL - European Computer Vision Library
* Version: 1.0.0
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#ifndef ECVL_TEST_CUDA_H_
#define ECVL_TEST_CUDA_H_

#include <stdint.h>

#define ECVL_TUPLE(type, ...) \
void RunTestCpuToGpuKernel##type(const uint8_t* data, uint8_t* res); \
void RunTestGpuToCpuKernel##type(uint8_t* data);
#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE


#endif // ECVL_TEST_CUDA_H_