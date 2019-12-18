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