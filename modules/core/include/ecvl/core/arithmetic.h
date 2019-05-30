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

// In-place Add between Images
void Add(Image& src1_dst, const Image& src2);
// Template implementation for in-place Add between Images
template <DataType a, DataType b>
struct StructAdd {
    static void actual_function(Image& src1_dst, const Image& src2) {
        using dsttype = typename TypeInfo<a>::basetype;

        View<a> vsrc1_dst(src1_dst);
        ConstView<b> vsrc2(src2);
        auto is1 = vsrc1_dst.Begin(), es1 = vsrc1_dst.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            *is1 = static_cast<dsttype>(*is1 + *is2);
        }
    }
};

// In-place Sub between Images
void Sub(Image& src1_dst, const Image& src2);
// Template implementation for in-place Sub between Images
template <DataType a, DataType b>
struct StructSub {
    static void actual_function(Image& src1_dst, const Image& src2) {
        using dsttype = typename TypeInfo<a>::basetype;

        View<a> vsrc1_dst(src1_dst);
        ConstView<b> vsrc2(src2);
        auto is1 = vsrc1_dst.Begin(), es1 = vsrc1_dst.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            *is1 = static_cast<dsttype>(*is1 - *is2);
        }
    }
};

/** @brief Template struct for in-place multiplication between images 
of any ecvl::DataType.

*/
void Mul(Image& src1_dst, const Image& src2);
// Template implementation for in-place Mul between Images
template <DataType a, DataType b>
struct StructMul {
    static void actual_function(Image& src1_dst, const Image& src2) {
        using dsttype = typename TypeInfo<a>::basetype;

        View<a> vsrc1_dst(src1_dst);
        ConstView<b> vsrc2(src2);
        auto is1 = vsrc1_dst.Begin(), es1 = vsrc1_dst.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            *is1 = static_cast<dsttype>(*is1 * *is2);
        }
    }
};

// In-place Div between Images
void Div(Image& src1_dst, const Image& src2);
// Template implementation for in-place Div between Images
template <DataType a, DataType b>
struct StructDiv {
    static void actual_function(Image& src1_dst, const Image& src2) {
        using dsttype = typename TypeInfo<a>::basetype;

        View<a> vsrc1_dst(src1_dst);
        ConstView<b> vsrc2(src2);
        auto is1 = vsrc1_dst.Begin(), es1 = vsrc1_dst.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            *is1 = static_cast<dsttype>(*is1 / *is2);
        }
    }
};


/** @brief Template specialization of the in-place multiplication
    function. In most cases it is better to use the @ref Mul.

    @param[in] img Image to be multiplied by a scalar value.
    @param[in] d Scalar value to use for the multiplication.
    @param[in] saturate Whether to apply saturation or not.

    @return Image containing the result of the multiplication, same as the input one.
*/
template<typename ViewType>
Image& Mul(Image& img, double d, bool saturate)
{
    ViewType v(img);
    auto i = v.Begin(), e = v.End();
    for (; i != e; ++i) {
        auto& p = *i;
        if (saturate) {
            p = saturate_cast<typename ViewType::basetype>(p * d);
        }
        else {
            p = static_cast<typename ViewType::basetype>(p * d);
        }
    }
    return img;
}

/** @brief In-place multiplication between an Image and a scalar value.
Without type promotion. @anchor Mul

The Mul() function multiplies an input image by a scalar value and stores
the result in the same image. The type of the image will not change. By
default a saturation will be applied. If it is not the desired behavior
change the "saturate" parameter to false.

@param[in] img Image to be multiplied (in-place) by a scalar value.
@param[in] d Scalar value to use for the multiplication.
@param[in] saturation Whether to apply saturation or not. Default is true.

@return Image containing the result of the multiplication.
*/
Image& Mul(Image& img, double d, bool saturate = true);

/** @brief Template specialization of the in-place sum
function. In most cases is better to use the @ref Sum.

@param[in] img Image to be summed by a scalar value.
@param[in] d Scalar value to use for the sum.
@param[in] saturate Whether to apply saturation or not.

@return Image containing the result of the sum, same as the input one.
*/
template<typename ViewType, typename T>
Image& Sum(Image& img, T value, bool saturate)
{
    ViewType v(img);
    auto i = v.Begin(), e = v.End();
    for (; i != e; ++i) {
        auto& p = *i;
        if (saturate) {
            p = saturate_cast<typename ViewType::basetype>(p + value);
        }
        else {
            p = static_cast<typename ViewType::basetype>(p + value);
        }
    }
    return img;
}

/** @brief In-place sum between an Image and a scalar value, without type
promotion. @anchor Sum

The Sum() function sum an input image by a scalar value and stores
the result in the same image. The type of the image will not change. By
default a saturation will be applied. If it is not the desired behavior
change the "saturate" parameter to false.

@param[in] img Image to be summed (in-place) by a scalar value.
@param[in] d Scalar value to use for the sum.
@param[in] saturation Whether to apply saturation or not. Default is true.

@return Image containing the result of the sum.
*/
template <typename T>
Image& Sum(Image& img, T value, bool saturate = true) {
    if (img.contiguous_) {
        switch (img.elemtype_)
        {
#define ECVL_TUPLE(name, ...) case DataType::name: return Sum<ContiguousView<DataType::name>>(img, value, saturate);
#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE
        default:
            throw std::runtime_error("How did you get here?");
        }
    }
    else {
        switch (img.elemtype_)
        {
#define ECVL_TUPLE(name, ...) case DataType::name: return Sum<View<DataType::name>>(img, value, saturate);
#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE
        default:
            throw std::runtime_error("How did you get here?");
        }
    }
}

} // namespace ecvl

#endif // !ECVL_ARITHMETIC_H_

