/*
* ECVL - European Computer Vision Library
* Version: 0.3.4
* copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include "test_cuda.h"

#include <stdint.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "ecvl/core/datatype.h"

using namespace ecvl;


#define ECVL_TUPLE(type, ...) \
__global__ void TestCpuToGpuKernel##type(const uint8_t* data, uint8_t* res) \
{ \
    using type_t = TypeInfo_t<DataType::type>; \
    const type_t* cur_data = reinterpret_cast<const type_t*>(data); \
    *res = 1; \
    if (*(cur_data++) != 50)    *res = 0; \
    if (*(cur_data++) != 32)    *res = 0; \
    if (*(cur_data++) != 14)    *res = 0; \
    if (*(cur_data++) != 60)    *res = 0; \
} \
\
void RunTestCpuToGpuKernel##type(const uint8_t* data, uint8_t* res) \
{ \
    TestCpuToGpuKernel##type<<<1,1>>>(data, res); \
} \
\
__global__ void TestGpuToCpuKernel##type(uint8_t* data) { \
    using type_t = TypeInfo_t<DataType::type>; \
    type_t* cur_data = reinterpret_cast<type_t*>(data); \
    *(cur_data++) = 50; \
    *(cur_data++) = 32; \
    *(cur_data++) = 14; \
    *(cur_data++) = 60; \
} \
\
void RunTestGpuToCpuKernel##type(uint8_t* data) { \
    TestGpuToCpuKernel##type<<<1,1>>>(data); \
}
#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE