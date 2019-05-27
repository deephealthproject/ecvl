#pragma once

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

