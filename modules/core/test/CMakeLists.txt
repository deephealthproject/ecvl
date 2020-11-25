# ECVL - European Computer Vision Library
# Version: 0.3.0
# copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
# Authors:
#    Costantino Grana (costantino.grana@unimore.it)
#    Federico Bolelli (federico.bolelli@unimore.it)
#    Michele Cancilla (michele.cancilla@unimore.it)
#    Laura Canalini (laura.canalini@unimore.it)
#    Stefano Allegretti (stefano.allegretti@unimore.it)
# All rights reserved.

target_sources(ECVL_TESTS PRIVATE
        test_core.cpp
        test_imgproc.cpp
        test_volproc.cpp
)

if(ECVL_GPU_AVAILABLE)
    target_sources(ECVL_CORE PRIVATE
        test_cuda.cu
        test_cuda.h
    )
endif()