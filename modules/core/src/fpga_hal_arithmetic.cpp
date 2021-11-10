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

#include "ecvl/core/fpga_hal.h"

#include "ecvl/core/image.h"
#include "ecvl/core/saturate_cast.h"

namespace ecvl
{

void FpgaHal::Neg(const Image& src, Image& dst, DataType dst_type, bool saturate)
{
    printf("FpgaHal::Neg not implemented\n"); exit(1);
}

void FpgaHal::Add(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate)
{
    printf("FpgaHal::Add not implemented\n"); exit(1);
}

void FpgaHal::Sub(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate)
{
    printf("FpgaHal::Sub not implemented\n"); exit(1);
}

void FpgaHal::Mul(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate)
{
    printf("FpgaHal::Mul not implemented\n"); exit(1);
}

void FpgaHal::Div(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate)
{
    printf("FpgaHal::Div not implemented\n"); exit(1);
}

#define ECVL_TUPLE(name, size, type, ...) \
void FpgaHal::Add(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) { printf("FpgaHal::Add not implemented\n"); exit(1); } \
void FpgaHal::Add(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { printf("FpgaHal::Add not implemented\n"); exit(1); } \
                                                                                                                                               \
void FpgaHal::Sub(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) { printf("FpgaHal::Sub not implemented\n"); exit(1); } \
void FpgaHal::Sub(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { printf("FpgaHal::Sub not implemented\n"); exit(1); } \
                                                                                                                                               \
void FpgaHal::Mul(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) { printf("FpgaHal::Mul not implemented\n"); exit(1); } \
void FpgaHal::Mul(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { printf("FpgaHal::Mul not implemented\n"); exit(1); } \
                                                                                                                                               \
void FpgaHal::Div(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) { printf("FpgaHal::Div not implemented\n"); exit(1); } \
void FpgaHal::Div(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { printf("FpgaHal::Div not implemented\n"); exit(1); } \
                                                                                                                                               \
void FpgaHal::SetTo(Image& src, type value) { printf("FpgaHal::SetTo not implemented\n"); exit(1); }                                                                          \

#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE


} // namespace ecvl