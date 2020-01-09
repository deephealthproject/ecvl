#include "ecvl/core/datatype.h"

namespace ecvl {

double SqDist(const Point2i& a, const Point2i& b) {
    return (a[0] - b[0])*(a[0] - b[0]) + (a[1] - b[1])*(a[1] - b[1]);
}

static uint8_t aDataTypeSize[] = {
#define ECVL_TUPLE(name, size, ...) size, 
#include "ecvl/core/datatype_tuples.inc.h"
#undef ECVL_TUPLE
};

uint8_t DataTypeSize(DataType dt) {
    return aDataTypeSize[static_cast<int>(dt)];
}

} // namespace ecvl