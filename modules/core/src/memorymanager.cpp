#include "memorymanager.h"

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
