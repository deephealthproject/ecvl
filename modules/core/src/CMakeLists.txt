# ECVL - European Computer Vision Library
# Version: 0.3.4
# copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
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
        cpu_hal_volproc.cpp
        datatype.cpp
        fpga_hal.cpp
        hal.cpp
        image.cpp
        imgcodecs.cpp
        imgproc.cpp
        support_opencv.cpp
        support_nifti.cpp
	fpga_hal_arithmetic.cpp
	fpga_hal_imgproc.cpp
	fpga_hal_volproc.cpp
)


if(ECVL_GPU_AVAILABLE)

    target_sources(ECVL_CORE
        PRIVATE

            cuda/gpu_hal.cpp

    )

endif()
