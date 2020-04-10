#include "ecvl/core/fpga_hal.h"

namespace ecvl
{

FpgaHal* FpgaHal::GetInstance()
{
#ifndef ECVL_FPGA
    ECVL_ERROR_DEVICE_UNAVAILABLE(FPGA)
#endif // ECVL_FPGA

    static FpgaHal instance; 	// Guaranteed to be destroyed.
                               // Instantiated on first use.
    return &instance;
}

} // namespace ecvl
