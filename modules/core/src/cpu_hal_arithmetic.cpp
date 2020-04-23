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

#include "ecvl/core/cpu_hal.h"

#include "ecvl/core/image.h"
#include "ecvl/core/saturate_cast.h"

namespace ecvl
{
// Struct template specialization of the negation of an Image.
template <DataType DT1, DataType DT2>
struct StructNegII
{
    static void _(const Image& src, Image& dst, bool saturate)
    {
        using dsttype = typename TypeInfo<DT2>::basetype;
        ConstView<DT1> vsrc(src);
        View<DT2> vdst(dst);
        auto is = vsrc.Begin(), es = vsrc.End();
        auto id = vdst.Begin();
        for (; is != es; ++is, ++id) {
            if (saturate) {
                *id = saturate_cast<dsttype>(-*is);
            }
            else {
                *id = static_cast<dsttype>(-*is);
            }
        }
    }
};

// CPU general implementation of the negation function.
void CpuHal::Neg(const Image& src, Image& dst, DataType dst_type, bool saturate)
{
    const Image* ptr = &src;
    Image tmp;
    if (&src == &dst) {
        tmp = src;
        ptr = &tmp;
    }

    // New datatype
    DataType datatype = dst_type;
    if (dst_type == DataType::none) {
        if (dst.IsEmpty()) {
            datatype = src.elemtype_;
        }
        else {
            datatype = dst.elemtype_;
        }
    }

    dst.Create(src.dims_, datatype, src.channels_, src.colortype_, src.spacings_, src.dev_);
    static constexpr SignedTable2D<StructNegII> table;
    table(src.elemtype_, dst.elemtype_)(*ptr, dst, saturate);
}

// Template implementation for the in-place Addition between Image(s)
template <DataType DT1, DataType DT2>
struct StructAddII
{
    static void _(Image& src1, const Image& src2, bool saturate)
    {
        using dsttype = typename TypeInfo<DT1>::basetype;
        View<DT1> vsrc1(src1);
        ConstView<DT2> vsrc2(src2);
        auto is1 = vsrc1.Begin(), es1 = vsrc1.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            if (saturate) {
                *is1 = saturate_cast<dsttype>(PromoteAdd(*is1, *is2));
            }
            else {
                *is1 = static_cast<dsttype>(*is1 + *is2);
            }
        }
    }
};

// CPU general implementation of the addition function.
void CpuHal::Add(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate)
{
    ecvl::CopyImage(src1, dst, dst_type);
    static constexpr Table2D<StructAddII> table;
    table(dst.elemtype_, src2.elemtype_)(dst, src2, saturate);
}

// Template implementation for the in-place subtraction between Image(s)
template <DataType DT1, DataType DT2>
struct StructSubII
{
    static void _(Image& src1, const Image& src2, bool saturate)
    {
        using dsttype = typename TypeInfo<DT1>::basetype;

        View<DT1> vsrc1(src1);
        ConstView<DT2> vsrc2(src2);
        auto is1 = vsrc1.Begin(), es1 = vsrc1.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            if (saturate) {
                *is1 = saturate_cast<dsttype>(PromoteSub(*is1, *is2));
            }
            else {
                *is1 = static_cast<dsttype>(*is1 - *is2);
            }
        }
    }
};

// CPU general implementation of the subtraction function.
void CpuHal::Sub(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate)
{
    const Image* ptr = &src2;
    Image tmp;
    if (&src2 == &dst) {
        tmp = src2;
        ptr = &tmp;
    }

    ecvl::CopyImage(src1, dst, dst_type);
    static constexpr Table2D<StructSubII> table;
    table(dst.elemtype_, src2.elemtype_)(dst, *ptr, saturate);
}

// Template implementation for the in-place Multiplication between Image(s)
template <DataType DT1, DataType DT2>
struct StructMulII
{
    static void _(Image& src1, const Image& src2, bool saturate)
    {
        using dsttype = typename TypeInfo<DT1>::basetype;

        View<DT1> vsrc1(src1);
        ConstView<DT2> vsrc2(src2);
        auto is1 = vsrc1.Begin(), es1 = vsrc1.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            if (saturate) {
                *is1 = saturate_cast<dsttype>(PromoteMul(*is1, *is2));
            }
            else {
                *is1 = static_cast<dsttype>(*is1 * *is2);
            }
        }
    }
};

// CPU general implementation of the multiplication function.
void CpuHal::Mul(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate)
{
    ecvl::CopyImage(src1, dst, dst_type);
    static constexpr Table2D<StructMulII> table;
    table(dst.elemtype_, src2.elemtype_)(dst, src2, saturate);
}

// Template implementation for the in-place division between Image(s)
template <DataType DT1, DataType DT2>
struct StructDivII
{
    static void _(Image& src1, const Image& src2, bool saturate)
    {
        using dsttype = typename TypeInfo<DT1>::basetype;

        View<DT1> vsrc1(src1);
        ConstView<DT2> vsrc2(src2);
        auto is1 = vsrc1.Begin(), es1 = vsrc1.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            if (saturate) {
                *is1 = saturate_cast<dsttype>(PromoteDiv(*is1, (*is2)));
            }
            else {
                *is1 = static_cast<dsttype>(*is1 / (*is2));
            }
        }
    }
};

// CPU general implementation of the division function.
void CpuHal::Div(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate)
{
    const Image* ptr = &src2;
    Image tmp;
    if (&src2 == &dst) {
        tmp = src2;
        ptr = &tmp;
    }

    ecvl::CopyImage(src1, dst, dst_type);
    static constexpr Table2D<StructDivII> table;
    table(dst.elemtype_, src2.elemtype_)(dst, *ptr, saturate);
}

// In-place addition between Image and scalar.
template<DataType DT, typename T>
struct StructAddIS
{
    static void _(Image& img, T value, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(PromoteAdd(p, value));
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(p + value);
            }
        }
    }
};
template<typename T>
static void AddImpl(const Image& src1, T src2, Image& dst, DataType dst_type, bool saturate)
{
    CopyImage(src1, dst);
    static constexpr Table1D<StructAddIS, T> table;
    table(dst.elemtype_)(dst, src2, saturate);
}
template<typename T>
static void AddImpl(T src1, const Image& src2, Image& dst, DataType dst_type, bool saturate)
{
    AddImpl(src2, src1, dst, dst_type, saturate);
}

// In-place subtraction between Image and scalar.
template<DataType DT, typename T>
struct StructSubIS
{
    static void _(Image& img, T value, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(PromoteSub(p, value));
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(p - value);
            }
        }
    }
};
template<DataType DT, typename T>
struct StructSubSI
{
    static void _(T value, Image& img, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(PromoteSub(value, p));
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(value - p);
            }
        }
    }
};
template<typename T>
static void SubImpl(const Image& src1, T src2, Image& dst, DataType dst_type, bool saturate)
{
    CopyImage(src1, dst);
    static constexpr Table1D<StructSubIS, T> table;
    table(dst.elemtype_)(dst, src2, saturate);
}
template<typename T>
static void SubImpl(T src1, const Image& src2, Image& dst, DataType dst_type, bool saturate)
{
    CopyImage(src2, dst);
    static constexpr Table1D<StructSubSI, T> table;
    table(dst.elemtype_)(src1, dst, saturate);
}

// In-place multiplication between Image and scalar.
template<DataType DT, typename T>
struct StructMulIS
{
    static void _(Image& img, T value, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(PromoteMul(p, value));
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(p * value);
            }
        }
    }
};
template<typename T>
static void MulImpl(const Image& src1, T src2, Image& dst, DataType dst_type, bool saturate)
{
    CopyImage(src1, dst);
    static constexpr Table1D<StructMulIS, T> table;
    table(dst.elemtype_)(dst, src2, saturate);
}
template<typename T>
static void MulImpl(T src1, const Image& src2, Image& dst, DataType dst_type, bool saturate)
{
    MulImpl(src2, src1, dst, dst_type, saturate);
}

// In-place division between Image and scalar.
template<DataType DT, typename T>
struct StructDivIS
{
    static void _(Image& img, T value, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(PromoteDiv(p, value));
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(p / value);
            }
        }
    }
};
template<DataType DT, typename T>
struct StructDivSI
{
    static void _(T value, Image& img, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(PromoteDiv(value, p));
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(value / p);
            }
        }
    }
};
template<typename T>
static void DivImpl(const Image& src1, T src2, Image& dst, DataType dst_type, bool saturate)
{
    CopyImage(src1, dst);
    static constexpr Table1D<StructDivIS, T> table;
    table(dst.elemtype_)(dst, src2, saturate);
}
template<typename T>
static void DivImpl(T src1, const Image& src2, Image& dst, DataType dst_type, bool saturate)
{
    CopyImage(src2, dst);
    static constexpr Table1D<StructDivSI, T> table;
    table(dst.elemtype_)(src1, dst, saturate);
}

#define ECVL_TUPLE(name, size, type, ...) \
void CpuHal::Add(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) { AddImpl(src1, src2, dst, dst_type, saturate); } \
void CpuHal::Add(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { AddImpl(src1, src2, dst, dst_type, saturate); } \
                                                                                                                                               \
void CpuHal::Sub(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) { SubImpl(src1, src2, dst, dst_type, saturate); } \
void CpuHal::Sub(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { SubImpl(src1, src2, dst, dst_type, saturate); } \
                                                                                                                                               \
void CpuHal::Mul(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) { MulImpl(src1, src2, dst, dst_type, saturate); } \
void CpuHal::Mul(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { MulImpl(src1, src2, dst, dst_type, saturate); } \
                                                                                                                                               \
void CpuHal::Div(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) { DivImpl(src1, src2, dst, dst_type, saturate); } \
void CpuHal::Div(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) { DivImpl(src1, src2, dst, dst_type, saturate); } \

#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE
} // namespace ecvl