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

#ifndef ECVL_CORE_H_
#define ECVL_CORE_H_

#include "core/arithmetic.h"
#include "core/datatype.h"
#include "core/image.h"
#include "core/imgcodecs.h"
#include "core/imgproc.h"
#include "core/iterators.h"
#include "core/macros.h"
#include "core/memorymanager.h"
#include "core/support_opencv.h"
#include "core/support_nifti.h"

#ifdef ECVL_WITH_DICOM
#include "core/support_dcmtk.h"
#endif

#ifdef ECVL_WITH_OPENSLIDE
#include "core/support_openslide.h"
#endif

#endif // ECVL_CORE_H_