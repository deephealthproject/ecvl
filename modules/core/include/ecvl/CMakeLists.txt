# ECVL - European Computer Vision Library
# Version: 0.2.1
# copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
# Authors:
#    Costantino Grana (costantino.grana@unimore.it)
#    Federico Bolelli (federico.bolelli@unimore.it)
#    Michele Cancilla (michele.cancilla@unimore.it)
#    Laura Canalini (laura.canalini@unimore.it)
#    Stefano Allegretti (stefano.allegretti@unimore.it)
# All rights reserved.

target_sources(ECVL_CORE
    PRIVATE
    
    core.h
    core/arithmetic.h
	core/arithmetic_impl.inc.h
	core/cpu_hal.h
    core/datatype.h
    core/datatype_existing_tuples.inc.h
    core/datatype_existing_tuples_signed.inc.h
    core/datatype_existing_tuples_unsigned.inc.h
    core/datatype_matrix.h
    core/datatype_tuples.inc.h
    core/fpga_hal.h
    core/hal.h
    core/image.h
    core/imgcodecs.h
    core/imgproc.h
    core/iterators.h
    core/iterators_impl.inc.h
	core/saturate_cast.h
    core/macros.h
    core/standard_errors.h
    core/support_opencv.h
    core/support_nifti.h
    core/type_promotion.h
)


if(ECVL_GPU_AVAILABLE)

    target_sources(ECVL_CORE
        PRIVATE
            
        core/cuda/gpu_hal.h
        core/cuda/common.h
    )

endif()