#ifndef ECVL_DATATYPE_H_
#define ECVL_DATATYPE_H_

#include <cstdint>

namespace ecvl {

/**  @brief DataType is an enum class which defines
data types allowed for images.
 
 @anchor DataType
 */
enum class DataType {
#define ECVL_TUPLE(name, ...) name,
#include "datatype_tuples.inc"
#undef ECVL_TUPLE
};

/**  @brief Provides the size in bytes of a given DataType.

Give one of the @ref DataType, the function returns its size in bytes.

@param[in] dt A DataType.

@return The DataType size in bytes
 */
uint8_t DataTypeSize(DataType dt);

/**  @brief 
 */
template<ecvl::DataType> struct TypeInfo { using basetype = void; };
#define ECVL_TUPLE(name, size, type, ...) template<> struct TypeInfo<ecvl::DataType::name> { using basetype = type; };
#include "datatype_tuples.inc"
#undef ECVL_TUPLE

} // namespace ecvl

#endif // ECVL_DATATYPE_H_
