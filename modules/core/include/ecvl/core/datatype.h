#ifndef ECVL_DATATYPE_H_
#define ECVL_DATATYPE_H_

#include <cstdint>
#include <limits>

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


} // namespace ecvl

#endif // ECVL_DATATYPE_H_
