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

target_sources(ECVL_GUI
    PRIVATE
    gui.cpp

)

if(${ECVL_WITH_OPENGL})
    target_sources(ECVL_GUI
    PRIVATE
    gui3D.cpp
)
endif(${ECVL_WITH_OPENGL})