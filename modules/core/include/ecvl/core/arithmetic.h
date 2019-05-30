#ifndef ECVL_ARITHMETIC_H_
#define ECVL_ARITHMETIC_H_

#include "ecvl/core/datatype_matrix.h"
#include "ecvl/core/image.h"

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

// In-place Add between Images TODO doc
void Add(Image& src1_dst, const Image& src2);
// Template implementation for in-place Add between Images
// TODO doc
template <DataType a, DataType b>
struct StructAdd {
    static void actual_function(Image& src1_dst, const Image& src2) {
        using dsttype = typename TypeInfo<a>::basetype;

        // TODO check before performing Add

        View<a> vsrc1_dst(src1_dst);
        ConstView<b> vsrc2(src2);
        auto is1 = vsrc1_dst.Begin(), es1 = vsrc1_dst.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            *is1 = static_cast<dsttype>(*is1 + *is2);
        }
    }
};

// In-place Sub between Images TODO doc
void Sub(Image& src1_dst, const Image& src2);
// Template implementation for in-place Sub between Images
// TODO doc
template <DataType a, DataType b>
struct StructSub {
    static void actual_function(Image& src1_dst, const Image& src2) {
        using dsttype = typename TypeInfo<a>::basetype;

        // TODO check before performing Add

        View<a> vsrc1_dst(src1_dst);
        ConstView<b> vsrc2(src2);
        auto is1 = vsrc1_dst.Begin(), es1 = vsrc1_dst.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            *is1 = static_cast<dsttype>(*is1 - *is2);
        }
    }
};

// In-place Mul between images TODO doc
void Mul(Image& src1_dst, const Image& src2);
/** @brief Template struct for in-place multiplication between images
of any ecvl::DataType. */
// TODO doc
template <DataType a, DataType b>
struct StructMul {
    static void actual_function(Image& src1_dst, const Image& src2) {
        using dsttype = typename TypeInfo<a>::basetype;

        // TODO check before performing Add

        View<a> vsrc1_dst(src1_dst);
        ConstView<b> vsrc2(src2);
        auto is1 = vsrc1_dst.Begin(), es1 = vsrc1_dst.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            *is1 = static_cast<dsttype>(*is1 * *is2);
        }
    }
};

// In-place Div between Images TODO doc
void Div(Image& src1_dst, const Image& src2);
// Template implementation for in-place Div between Images
// TODO doc
template <DataType a, DataType b>
struct StructDiv {
    static void actual_function(Image& src1_dst, const Image& src2) {
        using dsttype = typename TypeInfo<a>::basetype;

        // TODO check before performing Add

        View<a> vsrc1_dst(src1_dst);
        ConstView<b> vsrc2(src2);
        auto is1 = vsrc1_dst.Begin(), es1 = vsrc1_dst.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            *is1 = static_cast<dsttype>(*is1 / *is2);
        }
    }
};

// Struct template specialization of the in-place multiplication between Image and scalar. 
template<DataType DT, typename T>
struct StructScalarMul{
    static Image& ActualFunction(Image& img, T d, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(p * d);
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(p * d);
            }
        }
        return img;
    }
};

/** @brief In-place multiplication between an Image and a scalar value,
without type promotion. @anchor Mul

The Mul() function multiplies an input image by a scalar value and stores
the result in the same image. The type of the image will not change. By
default a saturation will be applied. If it is not the desired behavior
change the "saturate" parameter to false.

@param[in,out] img Image to be multiplied (in-place) by a scalar value.
@param[in] d Scalar value to use for the multiplication.
@param[in] saturate Whether to apply saturation or not. Default is true.

@return Reference to the Image containing the result of the multiplication.
*/
template<typename T>
Image& Mul(Image& img, T value, bool saturate = true) {                               

    // TODO add checks
    
    static constexpr Table1D<StructScalarMul, T> table;          
    return table(img.elemtype_)(img, value, saturate);
}

/** @overload [Image& Mul(Image& img, T value, bool saturate = true)] */
template<typename T>
Image& Mul(T value, Image& img, bool saturate = true) {                               
    return Mul(img, value, saturate);
}

// Struct template specialization of the in-place sum between Image and scalar. 
template<DataType DT, typename T>
struct StructScalarAdd{
    static Image& ActualFunction(Image& img, T value, bool saturate)
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
        return img;
    }
};

/** @brief In-place addition between an Image and a scalar value, without type
promotion. @anchor Add

The Add() function sums a scalar value to the input Image and stores
the result in the same image. The type of the image will not change. By
default a saturation will be applied. If it is not the desired behavior
change the "saturate" parameter to false.

@param[in,out] img Image to be summed (in-place) by a scalar value.
@param[in] value Scalar value to use for the sum.
@param[in] saturate Whether to apply saturation or not. Default is true.

@return Reference to the Image containing the result of the sum.
*/
template <typename T>
Image& Add(Image& img, T value, bool saturate = true) {
    
    // TODO add checks
    
    static constexpr Table1D<StructScalarAdd, T> table;          
    return table(img.elemtype_)(img, value, saturate);
}

/** @overload [Image& Sum(Image& img, T value, bool saturate = true)] */
template <typename T>
Image& Add(T value, Image& img, bool saturate = true) {
    return Add(img, value, saturate);
}

// Struct template specialization of the in-place subtraction between Image and scalar. 
template<DataType DT, typename T>
struct StructScalarSub{
    static Image& ActualFunction(Image& img, T value, bool saturate)
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
        return img;
    }
};

/** @brief In-place subtraction between an Image and a scalar value, without type
promotion. @anchor Sub

The Sub() function subtracts a scalar value from the input Image and stores
the result in the same image. The type of the image will not change. By
default a saturation will be applied. If it is not the desired behavior
change the "saturate" parameter to false.

@param[in,out] img Image to be subtracted (in-place) by a scalar value.
@param[in] value Scalar value to use for the subtraction.
@param[in] saturate Whether to apply saturation or not. Default is true.

@return Reference to the Image containing the result of the subtraction.
*/
template <typename T>
Image& Sub(Image& img, T value, bool saturate = true) {
    
    // TODO add checks

    static constexpr Table1D<StructScalarSub, T> table;          
    return table(img.elemtype_)(img, value, saturate);
}

// Struct template specialization of the in-place subtraction between a scalar value and an Image. 
template<DataType DT, typename T>
struct StructScalarSubInv{
    static Image& ActualFunction(T value, Image& img, bool saturate)
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
        return img;
    }
};

/** @brief In-place subtraction between a scalar value and an Image, without type
promotion. @anchor Sub

The Sub() function subtracts the input Image from a scalar value and stores
the result in the same image. The type of the image will not change. By
default a saturation will be applied. If it is not the desired behavior
change the "saturate" parameter to false.

@param[in] value Scalar value to use for the subtraction (Minuend).
@param[in,out] img Subtrahend of the operation. It will store the final result.
@param[in] saturate Whether to apply saturation or not. Default is true.

@return Reference to the Image containing the result of the subtraction.
*/
template <typename T>
Image& Sub(T value, Image& img, bool saturate = true) {
    
    // TODO add checks

    static constexpr Table1D<StructScalarSubInv, T> table;          
    return table(img.elemtype_)(value, img, saturate);
}

// Struct template specialization of the in-place division between Image and scalar. 
template<DataType DT, typename T>
struct StructScalarDiv{
    static Image& ActualFunction(Image& img, T value, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(p/value);
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(p/value);
            }
        }
        return img;
    }
};

/** @brief In-place division between an Image and a scalar value, without type
promotion. @anchor Div

The Div() function divides an input Image by a scalar value and stores
the result in the same image. The type of the image will not change. By
default a saturation will be applied. If it is not the desired behavior
change the "saturate" parameter to false.

@param[in,out] img Image to be divided (in-place) by a scalar value.
@param[in] value Scalar value to use for the division.
@param[in] saturate Whether to apply saturation or not. Default is true.

@return Reference to the Image containing the result of the division.
*/
template <typename T>
Image& Div(Image& img, T value, bool saturate = true) {
    
    // TODO add checks

    static constexpr Table1D<StructScalarDiv, T> table;          
    return table(img.elemtype_)(img, value, saturate);
}

// Struct template specialization of the in-place division between a scalar value and an Image. 
template<DataType DT, typename T, typename ET>
struct StructScalarDivInv{
    static Image& ActualFunction(T value, Image& img, bool saturate, ET epsilon)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(value/(p + epsilon));
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(value/(p + epsilon));
            }
        }
        return img;
    }
};

/** @brief In-place divion between a scalar value and an Image, without type
promotion. @anchor Div

The Div() function divides a scalar value by the input Image and stores
the result in the same image. The type of the image will not change. By
default a saturation will be applied. If it is not the desired behavior
change the "saturate" parameter to false.

@param[in] value Scalar value to use for the division (Dividend).
@param[in,out] img Divisor of the operation. It will store the final result.
@param[in] saturation Whether to apply saturation or not. Default is true.
@param[in] epsilon Small value to be added to the Image values before performing
            the division. If not specified by default it is the minimum positive
            number representable in a double.

@return Reference to the Image containing the result of the division.
*/
template <typename T, typename ET = double>
Image& Div(T value, Image& img, bool saturate = true, ET epsilon = std::numeric_limits<double>::min()) {

    // TODO add checks

    static constexpr Table1D<StructScalarDivInv, T, ET> table;          
    return table(img.elemtype_)(value, img, saturate, epsilon);
}

// Struct template specialization of the in-place negation of an Image. 
template<DataType DT>
struct StructScalarNeg{
    static Image& ActualFunction(Image& img)
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

/** @brief In-place negation of an Image. @anchor Neg

The Neg() function negates every value of an Image, and stores the 
the result in the same image. The type of the image will not change.

@param[in,out] img Image to be negated (in-place).

@return Reference to the Image containing the result of the negation.
*/
Image& Neg(Image& img) {
    
    // TODO add checks
    if (static_cast<size_t>(img.elemtype_) >= DataTypeSignedSize()) {
        throw std::runtime_error("Neg function is only allowed for signed images");
    }
    
    static constexpr SignedTable1D<StructScalarNeg> table;          
    return table(img.elemtype_)(img);
}

} // namespace ecvl

#endif // !ECVL_ARITHMETIC_H_

