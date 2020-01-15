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

#include "ecvl/core/memorymanager.h"

DefaultMemoryManager* DefaultMemoryManager::GetInstance()
{
    static DefaultMemoryManager instance;	// Guaranteed to be destroyed.
                                            // Instantiated on first use.
    return &instance;
}

ShallowMemoryManager* ShallowMemoryManager::GetInstance()
{
    static ShallowMemoryManager instance;	// Guaranteed to be destroyed.
                                            // Instantiated on first use.
    return &instance;
}
