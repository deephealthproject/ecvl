/*
* ECVL - European Computer Vision Library
* Version: 1.0.3
* copyright (c) 2021, UniversitÓ degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#ifndef ECVL_DATATYPE_H_
#define ECVL_DATATYPE_H_

#include <cstdint>
#include <cstddef>
#include <limits>
#include <array>
#include <vector>

namespace ecvl {

typedef std::array<int, 2> Point2i;
typedef std::array<double, 2> Point2d;
typedef std::array<int, 2> Size2i;
typedef std::array<double, 2> Size2d;
typedef std::vector<double> Scalar;

/** @brief Calculate the distance squared between two ecvl::Point2i.

@param[in] a First point integer coordinates.
@param[in] b Second point integer coordinates.
*/
double SqDist(const Point2i& a, const Point2i& b);

/**  @brief DataType is an enum class which defines
data types allowed for images.

 @anchor DataType
 */
enum class DataType {
#define ECVL_TUPLE(name, ...) name,
#include "datatype_tuples.inc.h"
#undef ECVL_TUPLE
};

/**  @brief Provides the size in bytes of a given DataType.

Given one of the @ref DataType, the function returns its size in bytes.

@param[in] dt A DataType.

@return The DataType size in bytes
 */
uint8_t DataTypeSize(DataType dt);

/**  @brief
 */
template<ecvl::DataType> struct TypeInfo { using basetype = void; };
#define ECVL_TUPLE(name, size, type, ...) template<> struct TypeInfo<ecvl::DataType::name> { using basetype = type; };
#include "datatype_tuples.inc.h"
#undef ECVL_TUPLE

template<ecvl::DataType DT>
using TypeInfo_t = typename TypeInfo<DT>::basetype;

/**  @brief Function to get the number of existing DataType at compile time.

@return The number of existing DataType.
 */
constexpr size_t DataTypeSize() {
    constexpr size_t size = 0
#define ECVL_TUPLE(name, ...) + 1
#include "datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE
        ;
    return size;
}

/**  @brief Function to get the number of existing signed DataType at compile time.

@return The number of existing signed DataType.
 */
constexpr size_t DataTypeSignedSize() {
    constexpr size_t size = 0
#define ECVL_TUPLE(name, ...) + 1
#include "datatype_existing_tuples_signed.inc.h"
#undef ECVL_TUPLE
        ;
    return size;
}

/**  @brief Function to get a std::array with all the DataType values at compile time.

@return A std::array with all the DataType values.

 */
constexpr std::array<DataType, DataTypeSize()> DataTypeArray() {
    //@cond
    constexpr std::array<DataType, DataTypeSize()> arr = {
#define ECVL_TUPLE(name, ...) DataType::name,
#include "datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE
    };
    return arr;
    //@endcond
}

/**  @brief Function to get a std::array with all the signed DataType values at compile time.

@return A std::array with all the signed DataType values.
 */
constexpr std::array<DataType, DataTypeSignedSize()> DataTypeSignedArray() {
    //@cond
    constexpr std::array<DataType, DataTypeSignedSize()> arr = {
#define ECVL_TUPLE(name, ...) DataType::name,
#include "datatype_existing_tuples_signed.inc.h"
#undef ECVL_TUPLE
    };
    return arr;
    //@endcond
}
} // namespace ecvl

#endif // ECVL_DATATYPE_H_
