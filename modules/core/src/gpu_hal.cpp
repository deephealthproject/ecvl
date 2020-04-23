#include "ecvl/core/gpu_hal.h"

namespace ecvl
{

GpuHal* GpuHal::GetInstance()
{
#ifndef ECVL_GPU
    ECVL_ERROR_DEVICE_UNAVAILABLE(GPU);
#endif // ECVL_GPU

    static GpuHal instance; 	// Guaranteed to be destroyed.
                                // Instantiated on first use.
    return &instance;
}

} // namespace ecvl
