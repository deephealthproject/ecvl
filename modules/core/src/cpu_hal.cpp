/*
* ECVL - European Computer Vision Library
* Version: 0.3.1
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
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
CpuHal* CpuHal::GetInstance()
{
    static CpuHal instance;	// Guaranteed to be destroyed.
                            // Instantiated on first use.
    return &instance;
}

// Copy Images of different DataTypes.
template<DataType SDT, DataType DDT>
struct StructCopyImage
{
    static void _(const Image& src, Image& dst)
    {
        using dsttype = typename TypeInfo<DDT>::basetype;

        ConstView<SDT> vsrc(src);
        View<DDT> vdst(dst);
        auto is = vsrc.Begin(), es = vsrc.End();
        auto id = vdst.Begin();
        for (; is != es; ++is, ++id) {
            *id = static_cast<dsttype>(*is);
        }
    }
};
void CpuHal::CopyImage(const Image& src, Image& dst)
{
    static constexpr Table2D<StructCopyImage> table;
    table(src.elemtype_, dst.elemtype_)(src, dst);
}

/** @brief Rearrange channels between Images of different DataTypes. */
template<DataType SDT, DataType DDT>
struct StructRearrangeImage
{
    static void _(const Image& src, Image& dst, const std::vector<int>& bindings)
    {
        using dsttype = typename TypeInfo<DDT>::basetype;
        using srctype = typename TypeInfo<SDT>::basetype;
        ConstView<SDT> vsrc(src);
        View<DDT> vdst(dst);
        auto id = vdst.Begin();

        for (size_t tmp_pos = 0; tmp_pos < dst.datasize_; tmp_pos += dst.elemsize_, ++id) {
            int x = static_cast<int>(tmp_pos);
            int src_pos = 0;
            for (int i = vsize(dst.dims_) - 1; i >= 0; i--) {
                src_pos += (x / dst.strides_[i]) * src.strides_[bindings[i]];
                x %= dst.strides_[i];
            }

            *id = static_cast<dsttype>(*reinterpret_cast<srctype*>(vsrc.data_ + src_pos));
        }
    }
};
void CpuHal::RearrangeChannels(const Image& src, Image& dst, const std::vector<int>& bindings)
{
    // TODO: checks?
    static constexpr Table2D<StructRearrangeImage> table;
    table(src.elemtype_, dst.elemtype_)(src, dst, bindings);
}

void ShallowCpuHal::Copy(const Image& src, Image& dst)
{
    // Copying from shallow -> destination becomes owner of the new data
    dst.hal_ = CpuHal::GetInstance();
    dst.hal_->Copy(src, dst);
}

ShallowCpuHal* ShallowCpuHal::GetInstance()
{
    static ShallowCpuHal instance;	// Guaranteed to be destroyed.
                                            // Instantiated on first use.
    return &instance;
}
} // namespace ecvl