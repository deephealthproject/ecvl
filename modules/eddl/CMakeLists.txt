# ECVL - European Computer Vision Library
# Version: 1.0.3
# copyright (c) 2021, UniversitÓ degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
# Authors:
#    Costantino Grana (costantino.grana@unimore.it)
#    Federico Bolelli (federico.bolelli@unimore.it)
#    Michele Cancilla (michele.cancilla@unimore.it)
#    Laura Canalini (laura.canalini@unimore.it)
#    Stefano Allegretti (stefano.allegretti@unimore.it)
# All rights reserved.

add_subdirectory("include/ecvl")
add_subdirectory(src)
if (${ECVL_TESTS})
    add_subdirectory(test)
endif()