/*
* ECVL - European Computer Vision Library
* Version: 0.3.4
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#define _USE_MATH_DEFINES
#include "ecvl/core/fpga_hal.h"

#include <iostream>

#include "ecvl/core/imgproc.h"
using namespace std;

namespace ecvl
{

    

void FpgaHal::SliceTimingCorrection(const Image& src, Image& dst, bool odd, bool down)
{
    printf("FpgaHal::SliceTimingCorrection not implemented\n"); exit(1);
}


} // namespace ecvl