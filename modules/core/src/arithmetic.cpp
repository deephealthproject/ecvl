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

#include "ecvl/core/arithmetic.h"

namespace ecvl {

/*********************************************************************************************/
/*  Arithmetic Operations Between Images and scalars (source Image is also the destination)  */
/*********************************************************************************************/
//TODO

/************************************************************************************/
/*  Arithmetic Operations Between Two Images (source and destination are the same)  */
/************************************************************************************/
// TODO add appropriate checks
#define STANDARD_INPLACE_OPERATION(Function, TemplateImplementation) \
void Function(Image& src1_dst, const Image& src2)                    \
{                                                                    \
    static constexpr Table2D<TemplateImplementation> table;          \
    table(src1_dst.elemtype_, src2.elemtype_)(src1_dst, src2);       \
}

/************************************************************************************/
/*   Unary Arithmetic Operations over Images (source and destination are the same)  */
/************************************************************************************/

// Struct template specialization of the in-place negation of an Image. 
template<DataType DT>
struct StructScalarNeg {
    static Image& _(Image& img)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            p = static_cast<typename TypeInfo<DT>::basetype>(-p);
        }
        return img;
    }
};

// TODO add appropriate checks
Image& Neg(Image& img) {

    // TODO add checks
    if (static_cast<size_t>(img.elemtype_) >= DataTypeSignedSize()) {
        ECVL_ERROR_NOT_ALLOWED_ON_UNSIGNED_IMG
    }

    static constexpr SignedTable1D<StructScalarNeg> table;
    return table(img.elemtype_)(img);
}

/************************************************************************************/
/*  Arithmetic Operations Between Two Images (source and destination are different) */
/************************************************************************************/
// TODO add appropriate checks
#define STANDARD_NON_INPLACE_OPERATION(Function)                                                  \
void Function(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) \
{                                                                                                 \
    if (src1.dims_ != src2.dims_ || src1.channels_ != src2.channels_) {                           \
        throw std::runtime_error("Source images must have the same dimensions and channels.");    \
    }                                                                                             \
                                                                                                  \
    if (!dst.IsOwner()) {                                                                         \
        if (src1.dims_ != dst.dims_ || src1.channels_ != dst.channels_) {                         \
            throw std::runtime_error("Non-owning data destination image must have the             \
                    same dimensions and channels as the sources.");                               \
        }                                                                                         \
    }                                                                                             \
                                                                                                  \
    CopyImage(src1, dst, dst_type);                                                               \
    Function(dst, src2);                                                                          \
}
//STANDARD_NON_INPLACE_OPERATION(Mul)
//STANDARD_NON_INPLACE_OPERATION(Add)
//STANDARD_NON_INPLACE_OPERATION(Sub)

void And(const Image& src1, const Image& src2, Image& dst) {

    if (src1.elemtype_ != DataType::uint8 || src1.colortype_ != ColorType::GRAY ||
        src2.elemtype_ != DataType::uint8 || src2.colortype_ != ColorType::GRAY) {
        ECVL_ERROR_WRONG_PARAMS("src images must have DataType::uint8 and ColorType::GRAY")
    }

    if (src1.dims_ != src2.dims_ || src1.channels_ != src2.channels_) {
        ECVL_ERROR_WRONG_PARAMS("src1 and src2 must have same dimensions and channels")
    }

    if (!dst.IsOwner()) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    Image tmp;
    bool use_tmp = false;
    if (&dst == &src1 || &dst == &src2) {
        use_tmp = true;
    }
    Image& img = use_tmp ? tmp : dst;
    img.Create(src1.dims_, DataType::uint8, src1.channels_, ColorType::GRAY, src1.spacings_);

    ConstView<DataType::uint8> view1(src1);
    ConstView<DataType::uint8> view2(src2);
    View<DataType::uint8> view_dst(img);

    auto it1 = view1.Begin();
    auto it2 = view2.Begin();
    auto it_dst = view_dst.Begin();
    auto it_dst_end = view_dst.End();

    for (; it_dst != it_dst_end; ++it_dst, ++it1, ++it2) {
        *it_dst = *it1 & *it2;
    }

    if (use_tmp) {
        dst = tmp;
    }
}

void Or(const Image& src1, const Image& src2, Image& dst) {

    if (src1.elemtype_ != DataType::uint8 || src1.colortype_ != ColorType::GRAY ||
        src2.elemtype_ != DataType::uint8 || src2.colortype_ != ColorType::GRAY) {
        ECVL_ERROR_WRONG_PARAMS("src images must have DataType::uint8 and ColorType::GRAY")
    }

    if (src1.dims_ != src2.dims_ || src1.channels_ != src2.channels_) {
        ECVL_ERROR_WRONG_PARAMS("src1 and src2 must have same dimensions and channels")
    }

    if (!dst.IsOwner()) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    Image tmp;
    bool use_tmp = false;
    if (&dst == &src1 || &dst == &src2) {
        use_tmp = true;
    }
    Image& img = use_tmp ? tmp : dst;
    img.Create(src1.dims_, DataType::uint8, src1.channels_, ColorType::GRAY, src1.spacings_);

    ConstView<DataType::uint8> view1(src1);
    ConstView<DataType::uint8> view2(src2);
    View<DataType::uint8> view_dst(img);

    auto it1 = view1.Begin();
    auto it2 = view2.Begin();
    auto it_dst = view_dst.Begin();
    auto it_dst_end = view_dst.End();

    for (; it_dst != it_dst_end; ++it_dst, ++it1, ++it2) {
        *it_dst = *it1 | *it2;
    }

    if (use_tmp) {
        dst = tmp;
    }
}

} // namespace ecvl