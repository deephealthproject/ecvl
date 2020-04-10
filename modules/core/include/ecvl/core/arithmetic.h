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

#ifndef ECVL_ARITHMETIC_H_
#define ECVL_ARITHMETIC_H_

#include <type_traits>

#include "ecvl/core/datatype_matrix.h"
#include "ecvl/core/image.h"
#include "ecvl/core/type_promotion.h"
#include "ecvl/core/standard_errors.h"

namespace ecvl
{

// TODO add appropriate checks (images' size, etc)                                                                     
#define BINARY_ARITHMETIC_OPERATION(Function)                                                                              \
inline void Function(const Image& src1, const Image& src2, Image& dst, DataType dst_type = DataType::none, bool saturate = true)  \
{                                                                                                                          \
    src1.hal_->Add(src1, src2, dst, dst_type, saturate);                                                                   \
}                                                                                                                          \
                                                                                                                           \
template<typename ST2>                                                                                                     \
void Function(const Image& src1, const ST2& src2, Image& dst, DataType dst_type = DataType::none, bool saturate = true)    \
{                                                                                                                          \
    src1.hal_->Add(src1, src2, dst, dst_type, saturate);                                                                   \
}                                                                                                                          \
                                                                                                                           \
template<typename ST1>                                                                                                     \
void Function(const ST1& src1, const Image& src2, Image& dst, DataType dst_type = DataType::none, bool saturate = true)    \
{                                                                                                                          \
    src2.hal_->Add(src1, src2, dst, dst_type, saturate);                                                                   \
}                                                                                                                          \

BINARY_ARITHMETIC_OPERATION(Add)
BINARY_ARITHMETIC_OPERATION(Sub)
BINARY_ARITHMETIC_OPERATION(Mul)
BINARY_ARITHMETIC_OPERATION(Div)

inline void Neg(const Image& src, Image& dst, DataType dst_type = DataType::none, bool saturate = true) {                                                                                                                          \
    src.hal_->Neg(src, dst, dst_type, saturate);
}

} // namespace ecvl

#endif // !ECVL_ARITHMETIC_H_