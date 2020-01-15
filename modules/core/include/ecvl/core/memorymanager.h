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

#ifndef ECVL_MEMORYMANAGER_H_
#define ECVL_MEMORYMANAGER_H_

#include <cstdint>
#include <cstring>
#include <stdexcept>

class MemoryManager {
public:
    virtual uint8_t* Allocate(size_t nbytes) = 0;
    virtual void Deallocate(uint8_t* data) = 0;
    virtual uint8_t* AllocateAndCopy(size_t nbytes, uint8_t* src) = 0;
    virtual ~MemoryManager() {}
};

class DefaultMemoryManager : public MemoryManager {
public:
    virtual uint8_t* Allocate(size_t nbytes) override {
        return new uint8_t[nbytes];
    }
    virtual void Deallocate(uint8_t* data) override {
        delete[] data;
    }
    virtual uint8_t* AllocateAndCopy(size_t nbytes, uint8_t* src) override {
        return reinterpret_cast<uint8_t*>(std::memcpy(new uint8_t[nbytes], src, nbytes));
    }

    static DefaultMemoryManager* GetInstance();
};

class ShallowMemoryManager : public MemoryManager {
public:
    virtual uint8_t* Allocate(size_t nbytes) override {
        throw std::runtime_error("ShallowMemoryManager cannot allocate memory");
    }
    virtual void Deallocate(uint8_t* data) override {}
    virtual uint8_t* AllocateAndCopy(size_t nbytes, uint8_t* src) override {
        return src;
    }

    static ShallowMemoryManager* GetInstance();
};

#endif // ECVL_MEMORYMANAGER_H_