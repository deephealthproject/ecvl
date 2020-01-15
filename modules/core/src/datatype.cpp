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