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

#ifndef ECVL_CPU_HAL_H_
#define ECVL_CPU_HAL_H_

#include <cstring>

#include "hal.h"

namespace ecvl
{

/** @brief CPU specific Hardware Abstraction Layer

*/
class CpuHal : public HardwareAbstractionLayer
{
public:
    uint8_t* MemAllocate(size_t nbytes) override
    {
        return new uint8_t[nbytes];
    }
    void MemDeallocate(uint8_t* data) override
    {
        delete[] data;
    }
    uint8_t* MemCopy(uint8_t* dst, const uint8_t* src, size_t nbytes) override
    {
        return reinterpret_cast<uint8_t*>(std::memcpy(dst, src, nbytes));
    }

    static CpuHal* GetInstance();

    void Neg(const Image& src, Image& dst, DataType dst_type, bool saturate) override;
    void Add(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override;
    void Sub(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override;
    void Mul(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override;
    void Div(const Image& src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override;

//#define ECVL_TUPLE(name, size, type, ...) \
//    void Add(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) override; \
//    void Add(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override; \
//                                                                                                                                 \
//    void Sub(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) override; \
//    void Sub(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override; \
//                                                                                                                                 \
//    void Mul(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) override; \
//    void Mul(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override; \
//                                                                                                                                 \
//    void Div(const Image& src1, type src2, Image& dst, DataType dst_type, bool saturate) override; \
//    void Div(type src1, const Image& src2, Image& dst, DataType dst_type, bool saturate) override; \
//
//#include "datatype_existing_tuples.inc.h"
//#undef ECVL_TUPLE

};

class ShallowCpuHal : public CpuHal
{
public:
    uint8_t* MemAllocate(size_t nbytes) override
    {
        throw std::runtime_error("ShallowCpuHal cannot allocate memory");
    }
    void MemDeallocate(uint8_t* data) override {}

    void Copy(const Image& src, Image& dst) override;

    bool IsOwner() const override { return false; };

    static ShallowCpuHal* GetInstance();
};

} // namespace ecvl
#endif // ECVL_CPU_HAL_H_