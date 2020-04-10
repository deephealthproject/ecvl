#include "ecvl/core/cpu_hal.h"

#include "ecvl/core/image.h"

namespace ecvl
{

CpuHal* CpuHal::GetInstance()
{
    static CpuHal instance;	// Guaranteed to be destroyed.
                            // Instantiated on first use.
    return &instance;
}

void ShallowCpuHal::Copy(const Image& src, Image& dst)
{
    // Copying from shallow -> destination becomes owner of the new data
    dst.hal_ = CpuHal::GetInstance();
    dst.hal_->Copy(src, dst);
}

ShallowCpuHal* ShallowCpuHal::GetInstance()
{
    static ShallowCpuHal instance;	// Guaranteed to be destroyed.
                                            // Instantiated on first use.
    return &instance;
}


} // namespace ecvl





