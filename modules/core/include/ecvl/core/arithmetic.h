#ifndef ECVL_ARITHMETIC_H_
#define ECVL_ARITHMETIC_H_

#include <type_traits>

#include "ecvl/core/datatype_matrix.h"
#include "ecvl/core/image.h"

#include "ecvl/core/standard_errors.h"

namespace ecvl {

/** @brief Saturate a value (of any type) to the specified type.

Given an input of any type the saturate_cast function provide an
output return value of the specified type applying saturation. When
the input value in greater than the maximum possible value (max) for
the output type, the max value is returned. When the input value in
lower than the minimum possible value (min) for the output type, the
min value is returned.

@param[in] v Input value (of any type).

@return Input value after cast and saturation.
*/
template<DataType ODT, typename IDT>
typename TypeInfo<ODT>::basetype saturate_cast(IDT v) {
    using basetype = typename TypeInfo<ODT>::basetype;

    if (v > std::numeric_limits<basetype>::max()) {
        return std::numeric_limits<basetype>::max();
    }
    if (v < std::numeric_limits<basetype>::min()) {
        return std::numeric_limits<basetype>::min();
    }

    return static_cast<basetype>(v);
}

/** @brief Saturate a value (of any type) to the specified type.

Given an input of any type the saturate_cast function provide an
output return value of the specified type applying saturation. When
the input value in greater than the maximum possible value (max) for
the output type, the max value is returned. When the input value in
lower than the minimum possible value (min) for the output type, the
min value is returned.

@param[in] v Input value (of any type).

@return Input value after cast and saturation.
*/
template<typename ODT, typename IDT>
ODT saturate_cast(const IDT& v) {

    if (v > std::numeric_limits<ODT>::max()) {
        return std::numeric_limits<ODT>::max();
    }
    if (v < std::numeric_limits<ODT>::min()) {
        return std::numeric_limits<ODT>::min();
    }

    return static_cast<ODT>(v);
}

/************************************************************************************/
/*   Unary Arithmetic Operations over Images (source and destination are the same)  */
/************************************************************************************/

/** @brief In-place negation of an Image. @anchor Neg

The Neg() function negates every value of an Image, and stores the
the result in the same image. The type of the image will not change.

@param[in,out] img Image to be negated (in-place).

@return Reference to the Image containing the result of the negation.
*/
Image& Neg(Image& img);

/************************************************************************************/
/*  Arithmetic Operations Between Two Images (source and destination are different) */
/************************************************************************************/

/** @brief Multiplies two Image(s) and stores the result in a third Image.

This procedure multiplies two Image(s) together and stores the result in
a third Image that will have the specified DataType. By default a saturation
will be applied. If it is not the desired behavior change the "saturate"
parameter to false.

@param[in] src1 Multiplier (first factor) Image.
@param[in] src2 Multiplicand (second factor) Image.
@param[out] dst Image into which save the result of the multiplication.
@param[in] dst_type DataType that destination Image must have at the end of the operation.
@param[in] saturate Whether to apply saturation or not. Default is true.

@return
*/
void Mul(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate = true);

/** @brief Subtracts two Image(s) and stores the result in a third Image.

This procedure subtracts the src2 Image from the src1 Image (src1 - src2) and stores the result
in the dst Image that will have the specified DataType. By default a saturation
will be applied. If it is not the desired behavior change the "saturate"
parameter to false.

@param[in] src1 Minuend Image.
@param[in] src2 Subtrahend Image.
@param[out] dst Image into which save the result of the division.
@param[in] dst_type DataType that destination Image must have at the end of the operation.
@param[in] saturate Whether to apply saturation or not. Default is true.

@return
*/
void Sub(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate = true);

/** @brief Adds two Image(s) and stores the result in a third Image.

This procedure adds src1 and src2 Image(s) (src1 + src2) and stores the result
in the dst Image that will have the specified DataType. By default a saturation
will be applied. If it is not the desired behavior change the "saturate"
parameter to false.

@param[in] src1 Augend (first addend) Image.
@param[in] src2 Addend (second addend) Image.
@param[out] dst Image into which save the result of the division.
@param[in] dst_type DataType that destination Image must have at the end of the operation.
@param[in] saturate Whether to apply saturation or not. Default is true.

@return
*/
void Add(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate = true);



/************************************************************************************/
/*  Addition                                                                        */
/************************************************************************************/

// Template implementation for the in-place Addition between Image(s)
template <DataType DT1, DataType DT2>
struct StructAdd {
    static void _(Image& src1, const Image& src2, bool saturate) {
        using dsttype = typename TypeInfo<DT1>::basetype;

        View<DT1> vsrc1(src1);
        ConstView<DT2> vsrc2(src2);
        auto is1 = vsrc1.Begin(), es1 = vsrc1.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            if (saturate) {
                *is1 = saturate_cast<dsttype>(*is1 + *is2);
            }
            else {
                *is1 = static_cast<dsttype>(*is1 + *is2);
            }
        }
    }
};

// Template specialization for the in-place Addition between Image and scalar. 
template<DataType DT, typename T>
struct ImageScalarAddImpl {
    static void _(Image& img, T value, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(p + value);
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(p + value);
            }
        }
    }
};

// Template non-specialized proxy for Add procedure (scalar + scalar)
template<typename ST1, typename ST2>
struct AddImpl {
    static void _(const ST1& src1, const ST2& src2, Image& dst, bool saturate) {

        // TODO add appropriate checks

        ECVL_ERROR_NOT_IMPLEMENTED
    }
};

// Template partial-specialized proxy for Add procedure (Image + scalar)
template<typename ST2>
struct AddImpl<Image, ST2> {
    static void _(const Image& src1, const ST2& src2, Image& dst, bool saturate) {

        // TODO add appropriate checks

        CopyImage(src1, dst);
        static constexpr Table1D<ImageScalarAddImpl, ST2> table;
        table(dst.elemtype_)(dst, src2, saturate);
    }
};

// Template partial-specialized proxy for Add procedure (scalar + Image)
template<typename ST1>
struct AddImpl<ST1, Image> {
    static void _(const ST1& src1, const Image& src2, Image& dst, bool saturate) {

        AddImpl<Image, ST1>::_(src2, src1, dst, saturate);
    }
};

// Template partial-specialized proxy for Add procedure (Image + Image)
template<>
struct AddImpl<Image, Image> {
    static void _(const Image& src1, const Image& src2, Image& dst, bool saturate) {

        // TODO add appropriate checks

        CopyImage(src1, dst);
        static constexpr Table2D<StructAdd> table;
        table(dst.elemtype_, src2.elemtype_)(dst, src2, saturate);
    }
};

/** @brief Adds two objects that could be either a scalar value or an Image,
storing the result into a destination Image. The procedure does not perform any type promotion.

The procedure takes two input values (src1 and src2) and adds them together,
storing the result into the destination image.
If one of the operands is an Image and the other one is a scalar value, each pixel of the Image is increased
by the scalar value, and the result is stored into dst.
If src1 and src2 are both Image(s) the pixel-wise addition is applied and, again,
the result is stored into dst.

Saturation is applied by default. If it is not the desired behavior change the
saturate parameter to false.

In any case, the operation performed is dst = src1 + src2.

@param[in] src1 Augend operand. Could be either a scalar or an Image.
@param[in] src2 Addend operand. Could be either a scalar or an Image.
@param[out] dst Destination Image. It will store the final result. If dst is not empty, its DataType will be preserved.
                Otherwise, it will have the same DataType as src1 if it is an Image, src2 otherwise.
@param[in] saturate Whether to apply saturation or not. Default is true.

@return.
*/
template<typename ST1, typename ST2>
void Add(const ST1& src1, const ST2& src2, Image& dst, bool saturate = true)
{
    AddImpl<ST1, ST2>::_(src1, src2, dst, saturate);
}


/************************************************************************************/
/*  Subtraction                                                                     */
/************************************************************************************/

// Template implementation for the in-place subtraction between Image(s)
template <DataType DT1, DataType DT2>
struct StructSub {
    static void _(Image& src1, const Image& src2, bool saturate) {
        using dsttype = typename TypeInfo<DT1>::basetype;

        View<DT1> vsrc1(src1);
        ConstView<DT2> vsrc2(src2);
        auto is1 = vsrc1.Begin(), es1 = vsrc1.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            if (saturate) {
                *is1 = saturate_cast<dsttype>(*is1 - *is2);
            }
            else {
                *is1 = static_cast<dsttype>(*is1 - *is2);
            }
        }
    }
};

// Template specialization for the in-place subtraction between Image and scalar. 
template<DataType DT, typename T>
struct ImageScalarSubImpl {
    static void _(Image& img, T value, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(p - value);
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(p - value);
            }
        }
    }
};

// Template specialization for the in-place subtraction between scalar and Image. 
template<DataType DT, typename T>
struct ScalarImageSubImpl {
    static void _(T value, Image& img, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(value - p);
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(value - p);
            }
        }
    }
};

// Template non-specialized proxy for Sub procedure (scalar - scalar)
template<typename ST1, typename ST2>
struct SubImpl {
    static void _(const ST1& src1, const ST2& src2, Image& dst, bool saturate) {

        // TODO add appropriate checks

        ECVL_ERROR_NOT_IMPLEMENTED
    }
};

// Template partial-specialized proxy for Sub procedure (Image - scalar)
template<typename ST2>
struct SubImpl<Image, ST2> {
    static void _(const Image& src1, const ST2& src2, Image& dst, bool saturate) {

        // TODO add appropriate checks

        CopyImage(src1, dst);
        static constexpr Table1D<ImageScalarSubImpl, ST2> table;
        table(dst.elemtype_)(dst, src2, saturate);
    }
};

// Template partial-specialized proxy for Sub procedure (scalar - Image)
template<typename ST1>
struct SubImpl<ST1, Image> {
    static void _(const ST1& src1, const Image& src2, Image& dst, bool saturate) {

        // TODO add appropriate checks

        CopyImage(src2, dst);
        static constexpr Table1D<ScalarImageSubImpl, ST1> table;
        table(dst.elemtype_)(src1, dst, saturate);
    }
};

// Template partial-specialized proxy for Sub procedure (Image - Image)
template<>
struct SubImpl<Image, Image> {
    static void _(const Image& src1, const Image& src2, Image& dst, bool saturate) {

        // TODO add appropriate checks

        CopyImage(src1, dst);
        static constexpr Table2D<StructSub> table;
        table(dst.elemtype_, src2.elemtype_)(dst, src2, saturate);
    }
};

/** @brief Subtracts two objects that could be either a scalar value or an Image,
storing the result into a destination Image. The procedure does not perform any type promotion.

The procedure takes two input values (src1 and src2) and subtracts the second from the
first, storing the result into the destination image.
If src1 is an Image and src2 is a scalar value, src2 is subtracted from all the pixels
inside src1 and the result is stored into dst.
If src1 is a scalar value and src2 is an Image, the opposite happens: src1 is diminished by
each pixel value of src2, and the result is stored into dst.
If src1 and src2 are both Image(s) the pixel-wise subtraction is applied and, again,
the result is stored into dst.

Saturation is applied by default. If it is not the desired behavior change the
saturate parameter to false.

In any case, the operation performed is dst = src1 - src2.

@param[in] src1 Minuend operand. Could be either a scalar or an Image.
@param[in] src2 Subtrahend operand. Could be either a scalar or an Image.
@param[out] dst Destination Image. It will store the final result. If dst is not empty, its DataType will be preserved.
                Otherwise, it will have the same DataType as src1 if it is an Image, src2 otherwise.
@param[in] saturate Whether to apply saturation or not. Default is true.

@return.
*/
template<typename ST1, typename ST2>
void Sub(const ST1& src1, const ST2& src2, Image& dst, bool saturate = true)
{
    SubImpl<ST1, ST2>::_(src1, src2, dst, saturate);
}



/************************************************************************************/
/*  Multiplication                                                                  */
/************************************************************************************/

// Template implementation for the in-place Multiplication between Image(s)
template <DataType DT1, DataType DT2>
struct StructMul {
    static void _(Image& src1, const Image& src2, bool saturate) {
        using dsttype = typename TypeInfo<DT1>::basetype;

        View<DT1> vsrc1(src1);
        ConstView<DT2> vsrc2(src2);
        auto is1 = vsrc1.Begin(), es1 = vsrc1.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            if (saturate) {
                *is1 = saturate_cast<dsttype>(*is1 * *is2);
            }
            else {
                *is1 = static_cast<dsttype>(*is1 * *is2);
            }
        }
    }
};

// Template specialization for the in-place Multiplication between Image and scalar. 
template<DataType DT, typename T>
struct ImageScalarMulImpl {
    static void _(Image& img, T value, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(p * value);
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(p * value);
            }
        }
    }
};

// Template non-specialized proxy for Mul procedure (scalar * scalar)
template<typename ST1, typename ST2>
struct MulImpl {
    static void _(const ST1& src1, const ST2& src2, Image& dst, bool saturate) {

        // TODO Mul appropriate checks

        ECVL_ERROR_NOT_IMPLEMENTED
    }
};

// Template partial-specialized proxy for Mul procedure (Image * scalar)
template<typename ST2>
struct MulImpl<Image, ST2> {
    static void _(const Image& src1, const ST2& src2, Image& dst, bool saturate) {

        // TODO Mul appropriate checks

        CopyImage(src1, dst);
        static constexpr Table1D<ImageScalarMulImpl, ST2> table;
        table(dst.elemtype_)(dst, src2, saturate);
    }
};

// Template partial-specialized proxy for Mul procedure (scalar * Image)
template<typename ST1>
struct MulImpl<ST1, Image> {
    static void _(const ST1& src1, const Image& src2, Image& dst, bool saturate) {

        MulImpl<Image, ST1>::_(src2, src1, dst, saturate);
    }
};

// Template partial-specialized proxy for Mul procedure (Image * Image)
template<>
struct MulImpl<Image, Image> {
    static void _(const Image& src1, const Image& src2, Image& dst, bool saturate) {

        // TODO Mul appropriate checks

        CopyImage(src1, dst);
        static constexpr Table2D<StructMul> table;
        table(dst.elemtype_, src2.elemtype_)(dst, src2, saturate);
    }
};

/** @brief Multiplies two objects that could be either a scalar value or an Image,
storing the result into a destination Image. The procedure does not perform any type promotion.

The procedure takes two input values (src1 and src2) and multiplies them together,
storing the result into the destination image.
If one of the operands is an Image and the other one is a scalar value, each pixel of the Image is 
multiplied by the scalar value, and the result is stored into dst.
If src1 and src2 are both Image(s) the pixel-wise multiplication is applied and, again,
the result is stored into dst.

Saturation is applied by default. If it is not the desired behavior change the
saturate parameter to false.

In any case, the operation performed is dst = src1 * src2.

@param[in] src1 Multiplier operand. Could be either a scalar or an Image.
@param[in] src2 Multiplicand operand. Could be either a scalar or an Image.
@param[out] dst Destination Image. It will store the final result. If dst is not empty, its DataType will be preserved.
                Otherwise, it will have the same DataType as src1 if it is an Image, src2 otherwise.
@param[in] saturate Whether to apply saturation or not. Default is true.

@return.
*/
template<typename ST1, typename ST2>
void Mul(const ST1& src1, const ST2& src2, Image& dst, bool saturate = true)
{
    MulImpl<ST1, ST2>::_(src1, src2, dst, saturate);
}



/************************************************************************************/
/*  Division                                                                        */
/************************************************************************************/

// Template implementation for the in-place division between Image(s)
template <DataType DT1, DataType DT2, typename ET>
struct StructDiv {
    static void _(Image& src1, const Image& src2, bool saturate, ET epsilon) {
        using dsttype = typename TypeInfo<DT1>::basetype;

        View<DT1> vsrc1(src1);
        ConstView<DT2> vsrc2(src2);
        auto is1 = vsrc1.Begin(), es1 = vsrc1.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            *is1 = static_cast<dsttype>(*is1 / (*is2 + epsilon));
        }
    }
};

// Template specialization for the in-place division between Image and scalar. 
template<DataType DT, typename T>
struct ImageScalarDivImpl {
    static void _(Image& img, T value, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(p / value);
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(p / value);
            }
        }
    }
};

// Template specialization for the in-place division between scalar and Image. 
template<DataType DT, typename T, typename ET>
struct ScalarImageDivImpl {
    static void _(T value, Image& img, bool saturate, ET epsilon)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(value / (p + epsilon));
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(value / (p + epsilon));
            }
        }
    }
};

// Template non-specialized proxy for Div procedure (scalar/scalar)
template<typename ST1, typename ST2, typename ET>
struct DivImpl {
    static void _(const ST1& src1, const ST2& src2, Image& dst, bool saturate, ET epsilon) {

        // TODO add appropriate checks

        ECVL_ERROR_NOT_IMPLEMENTED
    }
};

// Template partial-specialized proxy for Div procedure (Image/scalar)
template<typename ST2, typename ET>
struct DivImpl<Image, ST2, ET> {
    static void _(const Image& src1, const ST2& src2, Image& dst, bool saturate, ET epsilon) {

        // TODO add appropriate checks

        CopyImage(src1, dst);
        // TODO add appropriate checks
        if (src2 != 0) {
            static constexpr Table1D<ImageScalarDivImpl, ST2> table;
            table(dst.elemtype_)(dst, src2, saturate);
        }
        else {
            ECVL_ERROR_DIVISION_BY_ZERO
        }
    }
};

// Template partial-specialized proxy for Div procedure (scalar/Image)
template<typename ST1, typename ET>
struct DivImpl<ST1, Image, ET> {
    static void _(const ST1& src1, const Image& src2, Image& dst, bool saturate, ET epsilon) {

        // TODO add appropriate checks

        CopyImage(src2, dst);
        // TODO add appropriate checks
        static constexpr Table1D<ScalarImageDivImpl, ST1, ET> table;
        table(dst.elemtype_)(src1, dst, saturate, epsilon);
    }
};

// Template partial-specialized proxy for Div procedure (Image/Image)
template<typename ET>
struct DivImpl<Image, Image, ET> {
    static void _(const Image& src1, const Image& src2, Image& dst, bool saturate, ET epsilon) {

        // TODO add appropriate checks

        CopyImage(src1, dst);
        static constexpr Table2D<StructDiv, ET> table;
        table(dst.elemtype_, src2.elemtype_)(dst, src2, saturate, epsilon);
    }
};

/** @brief Divides two objects that could be either a scalar value or an Image,
storing the result into a destination Image. The procedure does not perform any type promotion.

The procedure takes two input values (src1 and src2) and divides the first by the
second, storing the result in the destination image. If src1 is an Image and src2 a
scalar value all the pixels inside src1 are divided by src2 and the result is
stored into dst. 
If src1 is a scalar value and src2 is an Image the opposite happens: all the pixel values
of src2 divide the scalar value src1 and the result is stored into dst.
If src1 and sr2 are both Image(s)
the pixel-wise division is applied and, again, the result is stored into dst.

Saturation is applied by default. If it is not the desired behavior change the
saturate parameter to false.

In the cases in which the divisor (denominator) is an Image an epsilon value is
summed to each divisor pixel value before the division in order to avoid divisions
by zero.

In any case, the operation performed is dst = src1 / src2.

@param[in] src1 Dividend (numerator) operand. Could be either a scalar or an Image.
@param[in] src2 Divisor (denominator) operand. Could be either a scalar or an Image.
@param[out] dst Destination Image. It will store the final result. If dst is not empty, its DataType will be preserved.
                Otherwise, it will have the same DataType as src1 if it is an Image, src2 otherwise.
@param[in] saturate Whether to apply saturation or not. Default is true.
@param[in] epsilon Small value to be added to divisor pixel values before performing
            the division. If not specified by default it is the minimum positive number
            representable in a double. It is ignored if src2 is a scalar value.

@return.
*/
template<typename ST1, typename ST2, typename ET = double>
void Div(const ST1& src1, const ST2& src2, Image& dst, bool saturate = true, ET epsilon = std::numeric_limits<double>::min())
{
    DivImpl<ST1, ST2, ET>::_(src1, src2, dst, saturate, epsilon);
}

/** @brief Divides two Image(s) and stores the result in a third Image.

This procedure divides the src1 Image by the src2 Image (src1/src2) and stores the result
into the dst Image that will have the specified DataType. By default a saturation
will be applied. If it is not the desired behavior change the "saturate"
parameter to false.

@param[in] src1 Dividend (numerator) Image.
@param[in] src2 Divisor (denominator) Image.
@param[out] dst Image into which save the result of the division.
@param[in] dst_type DataType that destination Image must have at the end of the operation.
@param[in] saturate Whether to apply saturation or not. Default is true.
@param[in] epsilon Small value to be added to the Image values before performing
            the division.If not specified by default it is the minimum positive
            number representable in a double.

@return
*/
//template <typename ET = double>
//void Div(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate = true, ET epsilon = std::numeric_limits<double>::min())
//{
//    // TODO add appropriate checks
//
//    if (src1.dims_ != src2.dims_ || src1.channels_ != src2.channels_) {
//        throw std::runtime_error("Source images must have the same dimensions and channels.");
//    }
//
//    if (!dst.IsOwner()) {
//        if (src1.dims_ != dst.dims_ || src1.channels_ != dst.channels_) {
//            throw std::runtime_error("Non-owning data destination image must have the same dimensions and channels as the sources.");
//        }
//    }
//
//    CopyImage(src1, dst, dst_type);
//    Div(dst, src2);
//}

} // namespace ecvl

#endif // !ECVL_ARITHMETIC_H_

