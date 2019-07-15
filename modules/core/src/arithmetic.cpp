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

} // namespace ecvl