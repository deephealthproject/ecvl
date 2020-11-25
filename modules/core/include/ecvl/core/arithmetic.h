/*
* ECVL - European Computer Vision Library
* Version: 0.3.0
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

// TODO add appropriate checks on the size of the images, color spaces and so on ...
#define BINARY_ARITHMETIC_OPERATION_IMAGE_IMAGE(Function)                                                                           \
inline void Function(const Image& src1, const Image& src2, Image& dst, DataType dst_type = DataType::none, bool saturate = true)    \
{                                                                                                                                   \
    src1.hal_->Function(src1, src2, dst, dst_type, saturate);                                                                       \
}                                                                                                                                   \

#define BINARY_ARITHMETIC_OPERATION_IMAGE_SCALAR(Function)                                                                          \
template<typename ST2>                                                                                                              \
void Function(const Image& src1, const ST2& src2, Image& dst, DataType dst_type = DataType::none, bool saturate = true)             \
{                                                                                                                                   \
    src1.hal_->Function(src1, src2, dst, dst_type, saturate);                                                                       \
}                                                                                                                                   \

#define BINARY_ARITHMETIC_OPERATION_SCALAR_IMAGE(Function)                                                                          \
template<typename ST1>                                                                                                              \
void Function(const ST1& src1, const Image& src2, Image& dst, DataType dst_type = DataType::none, bool saturate = true)             \
{                                                                                                                                   \
    src2.hal_->Function(src1, src2, dst, dst_type, saturate);                                                                       \
}                                                                                                                                   \

/** @brief Adds two source Images storing the result into a destination Image. 

    The procedure takes two input Images (src1 and src2) and adds them together performing a pixel-wise addition and
    storing the result into the destination image. Saturation is applied by default. If it is not the desired behavior 
    change the saturate parameter to false.

    The operation performed is dst = src1 + src2.

    If the DataType of the destination Image is not specified (DataType::none) the destination Image will preserve
    its own type, if any, or it will inherit the DataType of src1 otherwise.

    @param[in] src1 Augend operand (an Image).
    @param[in] src2 Addend operand (an Image).
    @param[out] dst Destination Image. It will store the final result. 
    @param[in] dst_type Desired type for the destination Image. If none (default) the destination
                        Image will preserve its own type, if any, or it will inherit the DataType 
                        of src1 otherwise.
    @param[in] saturate Whether to apply saturation or not. Default is true.

    @return.
*/
BINARY_ARITHMETIC_OPERATION_IMAGE_IMAGE(Add)

/** @brief Adds a scalar value to an Images storing the result into a destination Image.

    The procedure takes two input values, an Image (src1) and a scalar (src2) and adds them together: each pixel 
    of the Image is increased by the scalar value and the result is stored in the destination Image. Saturation is
    applied by default. If it is not the desired behavior change the saturate parameter to false.

    The operation performed is dst = src1 + src2.

    If the DataType of the destination Image is not specified (DataType::none) the destination Image will preserve
    its own type, if any, or it will inherit the DataType of src1 otherwise.

    @param[in] src1 Augend operand (an Image).
    @param[in] src2 Addend operand (a scalar value).
    @param[out] dst Destination Image. It will store the final result.
    @param[in] dst_type Desired type for the destination Image. If none (default) the destination
                        Image will preserve its own type, if any, or it will inherit the DataType
                        of src1 otherwise.
    @param[in] saturate Whether to apply saturation or not. Default is true.

    @return.
*/
BINARY_ARITHMETIC_OPERATION_IMAGE_SCALAR(Add)


/** @brief Adds a scalar value to an Images storing the result into a destination Image.

    The procedure takes two input values, a scalar (src1) and an Image (src2) and adds them together: each pixel
    of the Image is increased by the scalar value and the result is stored in the destination Image. Saturation is
    applied by default. If it is not the desired behavior change the saturate parameter to false.

    The operation performed is dst = src1 + src2.

    If the DataType of the destination Image is not specified (DataType::none) the destination Image will preserve
    its own type, if any, or it will inherit the DataType of src1 otherwise.

    @param[in] src1 Augend operand (a scalar value).
    @param[in] src2 Addend operand (an Image).
    @param[out] dst Destination Image. It will store the final result.
    @param[in] dst_type Desired type for the destination Image. If none (default) the destination
                        Image will preserve its own type, if any, or it will inherit the DataType
                        of src2 otherwise.
    @param[in] saturate Whether to apply saturation or not. Default is true.

    @return.
*/
BINARY_ARITHMETIC_OPERATION_SCALAR_IMAGE(Add)

BINARY_ARITHMETIC_OPERATION_IMAGE_IMAGE(Sub)
BINARY_ARITHMETIC_OPERATION_IMAGE_SCALAR(Sub)
BINARY_ARITHMETIC_OPERATION_SCALAR_IMAGE(Sub)

BINARY_ARITHMETIC_OPERATION_IMAGE_IMAGE(Mul)
BINARY_ARITHMETIC_OPERATION_IMAGE_SCALAR(Mul)
BINARY_ARITHMETIC_OPERATION_SCALAR_IMAGE(Mul)

BINARY_ARITHMETIC_OPERATION_IMAGE_IMAGE(Div)
BINARY_ARITHMETIC_OPERATION_IMAGE_SCALAR(Div)
BINARY_ARITHMETIC_OPERATION_SCALAR_IMAGE(Div)


/** @brief Negation of an Image. @anchor Neg

    The Neg() function negates every value of an Image, and stores the
    the result in a destination Image with the specified type.

    @param[in] src Image to be negated.
    @param[out] dst Destination Image. It will store the final result.
    @param[in] dst_type Desired type for the destination Image. If none (default) the destination
                        Image will preserve its own type, if any, or it will inherit the DataType
                        of src otherwise.
    @param[in] saturate Whether to apply saturation or not. Default is true.


    @return
*/
inline void Neg(const Image& src, Image& dst, DataType dst_type = DataType::none, bool saturate = true) {                                                                                                                          \
    src.hal_->Neg(src, dst, dst_type, saturate);
}

} // namespace ecvl

#endif // !ECVL_ARITHMETIC_H_