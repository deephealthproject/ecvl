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

#ifndef DATATYPE_MATRIX_H_
#define DATATYPE_MATRIX_H_

#include "datatype.h"

namespace ecvl {
// TODO internal doc
template<
    template<DataType DT, typename ...> class _StructFun,
    typename ...Args
>
struct Table1D {
    using fun_type = decltype(&_StructFun<static_cast<DataType>(0), Args...>::_);

    template<int i>
    struct integer {};

    template <int i>
    constexpr void FillData(integer<i>) {
        constexpr auto arr = DataTypeArray();
        data[i] = _StructFun<arr[i], Args...>::_;
        FillData(integer<i + 1>());
    }

    constexpr void FillData(integer<DataTypeSize()>) {}

    constexpr Table1D() : data() {
        FillData<0>(integer<0>());
    }

    inline fun_type operator()(DataType dt) const {
        return data[static_cast<int>(dt)];
    }

    fun_type data[DataTypeSize()];
};

// TODO internal doc
template<
    template<DataType, typename ...>class _StructFun,
    typename ...Args
>
struct SignedTable1D {
    using fun_type = decltype(&_StructFun<static_cast<DataType>(0), Args...>::_);

    template<int i>
    struct integer {};

    template <int i>
    constexpr void FillData(integer<i>) {
        constexpr auto arr = DataTypeSignedArray();
        data[i] = _StructFun<arr[i], Args...>::_;
        FillData(integer<i + 1>());
    }

    constexpr void FillData(integer<DataTypeSignedSize()>) {}

    constexpr SignedTable1D() : data() {
        FillData<0>(integer<0>());
    }

    inline fun_type operator()(DataType dt) const {
        return data[static_cast<int>(dt)];
    }

    fun_type data[DataTypeSignedSize()];
};

// TODO internal doc
template<
    template<DataType, DataType, typename ...>class _StructFun,
    typename ...Args
>
struct Table2D {
    using fun_type = decltype(&_StructFun<static_cast<DataType>(0), static_cast<DataType>(0), Args...>::_);

    template<int i>
    struct integer {};

    template <int i>
    constexpr void FillData(integer<i>) {
        constexpr auto arr = DataTypeArray();
        constexpr int src = i / DataTypeSize();
        constexpr int dst = i % DataTypeSize();
        data[i] = _StructFun<arr[src], arr[dst], Args...>::_;
        FillData(integer<i + 1>());
    }

    constexpr void FillData(integer< DataTypeSize()* DataTypeSize() >) {}

    constexpr Table2D() : data() {
        FillData<0>(integer<0>());
    }

    inline fun_type operator()(DataType src, DataType dst) const {
        int row = static_cast<int>(src);
        int col = static_cast<int>(dst);
        return data[row * DataTypeSize() + col];
    }

    fun_type data[DataTypeSize() * DataTypeSize()];
};

// TODO internal doc
template<
    template<DataType, DataType, typename ...>class _StructFun,
    typename ...Args
>
struct SignedTable2D {
    using fun_type = decltype(&_StructFun<static_cast<DataType>(0), static_cast<DataType>(0), Args...>::_);

    template<int i>
    struct integer {};

    template <int i>
    constexpr void FillData(integer<i>) {
        constexpr auto arr = DataTypeSignedArray();
        constexpr int src = i / DataTypeSignedSize();
        constexpr int dst = i % DataTypeSignedSize();
        data[i] = _StructFun<arr[src], arr[dst], Args...>::_;
        FillData(integer<i + 1>());
    }

    constexpr void FillData(integer< DataTypeSignedSize()* DataTypeSignedSize() >) {}

    constexpr SignedTable2D() : data() {
        FillData<0>(integer<0>());
    }

    inline fun_type operator()(DataType src, DataType dst) const {
        int row = static_cast<int>(src);
        int col = static_cast<int>(dst);
        return data[row * DataTypeSignedSize() + col];
    }

    fun_type data[DataTypeSignedSize() * DataTypeSignedSize()];
};
} // namespace ecvl

#endif // !DATATYPE_MATRIX_H_
