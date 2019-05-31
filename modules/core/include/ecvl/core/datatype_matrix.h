#ifndef DATATYPE_MATRIX_H_
#define DATATYPE_MATRIX_H_

#include "datatype.h"

namespace ecvl {

// TODO internal doc
template<template<DataType DT, typename ...>class _StructFun, typename ...Args>
struct Table1D {

    using fun_type = decltype(&_StructFun<static_cast<DataType>(0), Args...>::ActualFunction);

    template<int i>
    struct integer {};

    template <int i>
    constexpr void FillData(integer<i>) {
        constexpr auto arr = DataTypeArray();
        data[i] = _StructFun<arr[i], Args...>::ActualFunction;
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
template<template<DataType, typename ...>class _StructFun, typename ...Args>
struct SignedTable1D {

    using fun_type = decltype(&_StructFun<static_cast<DataType>(0), Args...>::ActualFunction);

    template<int i>
    struct integer {};

    template <int i>
    constexpr void FillData(integer<i>) {
        constexpr auto arr = DataTypeSignedArray();
        data[i] = _StructFun<arr[i], Args...>::ActualFunction;
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
template<template<DataType, DataType, typename ...>class _StructFun, typename ...Args>
struct Table2D {

    using fun_type = decltype(&_StructFun<static_cast<DataType>(0), static_cast<DataType>(0), Args...>::ActualFunction);

    template<int i>
    struct integer {};

    template <int i>
    constexpr void FillData(integer<i>) {
        constexpr auto arr = DataTypeArray();
        constexpr int src = i / DataTypeSize();
        constexpr int dst = i % DataTypeSize();
        data[i] = _StructFun<arr[src], arr[dst], Args...>::ActualFunction;
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
//template<template<DataType, DataType, DataType, typename ...>class _StructFun, typename ...Args>
//struct Table3D {
//
//    using fun_type = decltype(&_StructFun<static_cast<DataType>(0), static_cast<DataType>(0), static_cast<DataType>(0), Args...>::ActualFunction);
//    
//    template<int i>
//    struct integer {};
//    
//    template <int i>
//    constexpr void FillData(integer<i>) {
//        constexpr auto arr = DataTypeArray();
//        constexpr int src1 = (i / DataTypeSize()) % DataTypeSize(); // Row index in a plane
//        constexpr int src2 = i % DataTypeSize();                    // Col index in a row of a plane
//        constexpr int dst = i / (DataTypeSize() * DataTypeSize());  // Plane index
//        //data[i] = _StructFun<arr[src1], arr[src2], arr[dst], Args...>::ActualFunction;
//        FillData(integer<i + 1>());
//    }
//    
//    constexpr void FillData(integer< DataTypeSize() * DataTypeSize() * DataTypeSize() >) {}
//    
//    constexpr Table3D() : data() {
//        FillData(integer<0>());
//    }
//    
//    inline fun_type operator()(DataType src1, DataType src2, DataType dst) const {
//        int row = static_cast<int>(src1);
//        int col = static_cast<int>(src2);
//        int pla = static_cast<int>(dst);
//        return data[pla*(DataTypeSize() * DataTypeSize()) + row * DataTypeSize() + col];
//    }
//    
//    fun_type data[DataTypeSize() * DataTypeSize() * DataTypeSize()];
//};

} // namespace ecvl

#endif // !DATATYPE_MATRIX_H_

