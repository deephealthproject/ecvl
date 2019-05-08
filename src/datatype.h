#ifndef ECVL_DATATYPE_H_
#define ECVL_DATATYPE_H_

#include <map>

namespace ecvl {

// enum class DataType { uint8, uint16, ... };
enum class DataType {
#define TUPLE(name, size) name,
#include "datatype_tuples.inc"
#undef TUPLE
};

uint8_t DataTypeSize(DataType dt);

} // namespace ecvl

#endif // ECVL_DATATYPE_H_
