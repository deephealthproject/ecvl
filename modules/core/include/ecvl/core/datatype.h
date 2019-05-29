#ifndef ECVL_DATATYPE_H_
#define ECVL_DATATYPE_H_

#include <cstdint>
#include <limits>
#include <array>

namespace ecvl {

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


/**  @brief Function to get a std::array with all the DataType values at compile time.

@return A std::array with all the DataType values.
 */
constexpr std::array<DataType, DataTypeSize()> DataTypeArray() {
    constexpr std::array<DataType, DataTypeSize()> arr = {
#define ECVL_TUPLE(name, ...) DataType::name,
#include "datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE	
    };
    return arr;
}

} // namespace ecvl

#endif // ECVL_DATATYPE_H_
