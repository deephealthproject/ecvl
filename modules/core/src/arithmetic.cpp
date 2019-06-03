#include "ecvl/core/arithmetic.h"

namespace ecvl {

/*********************************************************************************************/
/*  Arithmetic Operations Between Images and scalars (source Image is also the destination)  */
/*********************************************************************************************/
//TODO

/************************************************************************************/
/*  Arithmetic Operations Between Two Images (source and destination are the same)  */
/************************************************************************************/
#define STANDARD_INPLACE_OPERATION(Function, TemplateImplementation) \
void Function(Image& src1_dst, const Image& src2)                    \
{                                                                    \
    static constexpr Table2D<TemplateImplementation> table;          \
    table(src1_dst.elemtype_, src2.elemtype_)(src1_dst, src2);       \
}
STANDARD_INPLACE_OPERATION(Add, StructAdd)
STANDARD_INPLACE_OPERATION(Sub, StructSub)
STANDARD_INPLACE_OPERATION(Mul, StructMul)
STANDARD_INPLACE_OPERATION(Div, StructDiv)

/************************************************************************************/
/*   Unary Arithmetic Operations over Images (source and destination are the same)  */
/************************************************************************************/
Image& Neg(Image& img) {

    // TODO add checks
    if (static_cast<size_t>(img.elemtype_) >= DataTypeSignedSize()) {
        throw std::runtime_error("Neg function is only allowed for signed images");
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
STANDARD_NON_INPLACE_OPERATION(Mul)
STANDARD_NON_INPLACE_OPERATION(Add)
STANDARD_NON_INPLACE_OPERATION(Sub)
STANDARD_NON_INPLACE_OPERATION(Div)

} // namespace ecvl