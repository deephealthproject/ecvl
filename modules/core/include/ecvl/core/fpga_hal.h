/*
* ECVL - European Computer Vision Library
* Version: 0.2.1
* copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#ifndef ECVL_FPGA_HAL_H_
#define ECVL_FPGA_HAL_H_

#include "hal.h"
#include "ecvl/core/image.h"
#include <iostream>

namespace ecvl
{
using namespace std;
class FpgaHal : public HardwareAbstractionLayer
{
public:
	
	uint8_t* MemAllocate(size_t nbytes) override
    {
		cout << "entraaaaaaaaa MemAllocate fpgaaa" << endl;
        return new uint8_t[nbytes];
    }
    void MemDeallocate(uint8_t* data) override
    {
		cout << "entraaaaaaaaa MemDeallocate fpgaaa" << endl;
        delete[] data;
		//free(data);
    }
    uint8_t* MemCopy(uint8_t* dst, const uint8_t* src, size_t nbytes) override
    {
		cout << "entraaaaaaaaa MemCopy fpgaaa" << endl;
        return static_cast<uint8_t*>(std::memcpy(dst, src, nbytes));
    }

    static FpgaHal* GetInstance();
	
	void CopyImage(const Image& src, Image& dst) override;
	void RearrangeChannels(const Image& src, Image& dst, const std::vector<int>& bindings) override;
	void FromCpu(Image& src) override;
    void ToCpu(Image& src) override;
	
	void ResizeDim(const ecvl::Image& src, ecvl::Image& dst, const std::vector<int>& newdims, InterpolationType interp) override;
    void ResizeScale(const Image& src, Image& dst, const std::vector<double>& scales, InterpolationType interp) override;
    void Flip2D(const ecvl::Image& src, ecvl::Image& dst) override;
    void Mirror2D(const ecvl::Image& src, ecvl::Image& dst) override;
	void ChangeColorSpace(const Image& src, Image& dst, ColorType new_type) override;
	void Threshold(const Image& src, Image& dst, double thresh, double maxval, ThresholdingType thresh_type) override;
	int OtsuThreshold(const Image& src) override;
	void GaussianBlur(const Image& src, Image& dst, int sizeX, int sizeY, double sigmaX, double sigmaY) override;
};
} // namespace ecvl
#endif // ECVL_FPGA_HAL_H_
