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

#include "../../../../../3rdparty/xfopencv/examples_sdaccel/resize/xcl2.hpp"

extern cl::Context context;
extern cl::CommandQueue q;
//extern size_t offset;

namespace ecvl
{
using namespace std;

void fpga_init();
void ReturnBuffer(cv::Mat& src,  ecvl::Image& dst);

class FpgaHal : public HardwareAbstractionLayer
{
public:
	
	
	uint8_t* MemAllocate(size_t nbytes) override
    {
		cout << "Inside MemAllocate fpga" << endl;
		cl::Buffer *imageToDevice = new cl::Buffer(context, CL_MEM_READ_ONLY, nbytes);
		return (uint8_t*) imageToDevice;
    }
    void MemDeallocate(uint8_t* data) override
    {
		cout << "Inside MemDeallocate fpga" << endl;
        delete[] data;
    }
    uint8_t* MemCopy(uint8_t* dst, const uint8_t* src, size_t nbytes) override
    {
		cout << "Inside MemCopy fpga" << endl;
		cl::Buffer *buffer_a = (cl::Buffer*) dst;
        return (uint8_t*) q.enqueueWriteBuffer(*buffer_a, CL_TRUE, 0, nbytes, src);
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
