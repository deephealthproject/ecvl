#include "datatype.h"

namespace ecvl {

static uint8_t aDataTypeSize[] = {
#define TUPLE(name, size, ...) size,
#include "datatype_tuples.inc"
#undef TUPLE
};

uint8_t DataTypeSize(DataType dt) {
    return aDataTypeSize[static_cast<int>(dt)];
}

}