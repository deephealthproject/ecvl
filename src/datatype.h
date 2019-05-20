#ifndef ECVL_DATATYPE_H_
#define ECVL_DATATYPE_H_

#include <cstdint>

namespace ecvl {

// enum class DataType { uint8, uint16, ... };
enum class DataType {
#define TUPLE(name, ...) name,
#include "datatype_tuples.inc"
#undef TUPLE
};

uint8_t DataTypeSize(DataType dt);

template<ecvl::DataType> struct TypeInfo { using basetype = void; };
#define TUPLE(name, size, type, ...) template<> struct TypeInfo<ecvl::DataType::##name> { using basetype = type; };
#include "datatype_tuples.inc"
#undef TUPLE

} // namespace ecvl

#endif // ECVL_DATATYPE_H_
