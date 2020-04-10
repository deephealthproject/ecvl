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
struct StructNeg
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

// TODO add appropriate checks
void CpuHal::Neg(const Image& src, Image& dst, DataType dst_type, bool saturate)
{
    const Image *ptr = &src;
    Image tmp;
    if (&src == &dst) {
        tmp = src;
        ptr = &tmp;
    }
    dst.Create(src.dims_, dst_type, src.channels_, src.colortype_, src.spacings_, src.dev_);
    static constexpr SignedTable2D<StructNeg> table;
    table(src.elemtype_, dst.elemtype_)(*ptr, dst, saturate);
}

// Template implementation for the in-place Addition between Image(s)
template <DataType DT1, DataType DT2>
struct StructAdd
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

void CpuHal::Add(const Image & src1, const Image & src2, Image & dst, DataType dst_type, bool saturate)
{
    CopyImage(src1, dst, dst_type);
    static constexpr Table2D<StructAdd> table;
    table(dst.elemtype_, src2.elemtype_)(dst, src2, saturate);
}

// Template implementation for the in-place subtraction between Image(s)
template <DataType DT1, DataType DT2>
struct StructSub
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

void CpuHal::Sub(const Image & src1, const Image & src2, Image & dst, DataType dst_type, bool saturate)
{
    const Image *ptr = &src2;
    Image tmp;
    if (&src2 == &dst) {
        tmp = src2;
        ptr = &tmp;
    }
    
    CopyImage(src1, dst, dst_type);
    static constexpr Table2D<StructAdd> table;
    table(dst.elemtype_, src2.elemtype_)(dst, *ptr, saturate);
}

// Template implementation for the in-place Multiplication between Image(s)
template <DataType DT1, DataType DT2>
struct StructMul
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

void CpuHal::Mul(const Image & src1, const Image & src2, Image & dst, DataType dst_type, bool saturate)
{
    CopyImage(src1, dst, dst_type);
    static constexpr Table2D<StructMul> table;
    table(dst.elemtype_, src2.elemtype_)(dst, src2, saturate);
}

// Template implementation for the in-place division between Image(s)
template <DataType DT1, DataType DT2>
struct StructDiv
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

void CpuHal::Div(const Image & src1, const Image & src2, Image & dst, DataType dst_type, bool saturate)
{
    const Image *ptr = &src2;
    Image tmp;
    if (&src2 == &dst) {
        tmp = src2;
        ptr = &tmp;
    }

    CopyImage(src1, dst, dst_type);
    static constexpr Table2D<StructDiv> table;
    table(dst.elemtype_, src2.elemtype_)(dst, *ptr, saturate);
}



} // namespace ecvl