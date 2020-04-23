# ECVL - European Computer Vision Library
# Version: 0.1
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
        arithmetic.cpp
        cpu_hal.cpp
        cpu_hal_arithmetic.cpp
        cpu_hal_imgproc.cpp
        datatype.cpp
        fpga_hal.cpp
        hal.cpp
        image.cpp
        imgcodecs.cpp
        imgproc.cpp
        support_opencv.cpp
        support_nifti.cpp
)


if(ECVL_GPU_AVAILABLE)

    target_sources(ECVL_CORE
        PRIVATE

            cuda/gpu_hal.cpp

)

endif()