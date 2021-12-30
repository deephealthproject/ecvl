/*
* ECVL - European Computer Vision Library
* Version: 0.3.4
* copyright (c) 2021, Universit� degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include <ecvl/core/image.h>
#include "ecvl/core/fpga_hal.h"

#include "ecvl/core/fpga_hal.h"
#include <ecvl/core/cpu_hal.h>
//#include "ecvl/core/imgproc_fpga.h"
#include <random>
#include <opencv2/photo.hpp>

#if OpenCV_VERSION_MAJOR >= 4
#include <opencv2/calib3d.hpp>
#endif // #if OpenCV_VERSION_MAJOR >= 4


#include "ecvl/core/image.h"
#include "ecvl/core/imgproc.h"
#include "ecvl/core/arithmetic.h"
#include "ecvl/core/support_opencv.h"
#include <functional>

#define CL_HPP_ENABLE_EXCEPTIONS
#include "ecvl/core/xcl2.hpp"      // OpenCL header
//#include <CL/cl2.hpp>

#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ecvl/core/datatype_matrix.h"
#include "ecvl/core/standard_errors.h"
#include <iostream>

cl::CommandQueue *q;
cl::Device device;
cl::Context *context;
cl::Program::Binaries bins;
cl::Program program;
std::vector<cl::Device> devices;
std::string device_name;
std::string binaryFile;

cl::Kernel kernel_otsu_threshold, kernel_threshold, kernel_mirror2d, kernel_flip2d, kernel_rgb_2_gray, kernel_gaussian_blur, kernel_resize, kernel_warp_transform, kernel_filter2d;

#define ECVL_FPGA

namespace ecvl
{

void fpga_init(){

  printf("  - FPGA: fpga_init\n");

  devices = xcl::get_xil_devices();
  device = devices[0];
  cl_int err;

  printf("    - device found\n");

  OCL_CHECK(err, context = new cl::Context(device, NULL, NULL, NULL, &err));

  printf("    - context created\n");

  OCL_CHECK(err, q = new cl::CommandQueue(*context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

  printf("    - command queue created\n");

  device_name = device.getInfo<CL_DEVICE_NAME>();
  std::string xcl_file = "ecvl_kernels.xclbin";
  auto fileBuf = xcl::read_binary_file(xcl_file);

  printf("    - binary file found\n");

  bins = cl::Program::Binaries{{fileBuf.data(), fileBuf.size()}};
  devices.resize(1);

  printf("    - binary file imported\n");

  OCL_CHECK(err, program = cl::Program(*context, devices, bins, NULL, &err));
  if (err != CL_SUCCESS) printf("Error creating program 3\n");
  printf("    - program created\n");

  OCL_CHECK(err, kernel_filter2d = cl::Kernel(program,"filter2d_accel", &err));
  printf("    - filter2d kernel created\n");

  //OCL_CHECK(err, kernel_warp_transform = cl::Kernel(program,"warpTransform_accel", &err));
  //printf("    - warp_transfrom kernel created\n");

  OCL_CHECK(err, kernel_resize = cl::Kernel(program,"resize_accel", &err));
  printf("    - resize kernel created\n");

  OCL_CHECK(err, kernel_gaussian_blur = cl::Kernel(program,"gaussian_accel", &err));
  printf("    - gaussian_blur kernel created\n");

  OCL_CHECK(err, kernel_rgb_2_gray = cl::Kernel(program,"rgb2gray_accel", &err));
  printf("    - rgb2gray kernel created\n");

  OCL_CHECK(err, kernel_flip2d = cl::Kernel(program,"flipvertical_accel", &err));
  printf("    - flip2d kernel created\n");

  OCL_CHECK(err, kernel_mirror2d = cl::Kernel(program,"mirror_accel", &err));
  printf("    - mirror2d kernel created\n");

  OCL_CHECK(err, kernel_threshold = cl::Kernel(program,"threshold_accel", &err));
  printf("    - threshold kernel created\n");

  OCL_CHECK(err, kernel_otsu_threshold = cl::Kernel(program,"otsuThreshold_accel", &err));
  printf("    - otsu_threshold kernel created\n");

  printf("END FPGA INIT\n");
}

void FpgaHal::FromCpu(Image& src)
{
    cl_int err;

    printf("  - FPGA: FromCpu\n");
    if (!src.contiguous_) {
        // The copy constructor creates a new contiguous image
        src = Image(src);
        printf("unexpected hola\n");
    }

    printf("    - Creating buffer\n");
    OCL_CHECK(err, src.fpga_buffer = new cl::Buffer(*context, CL_MEM_READ_WRITE, src.datasize_, NULL, &err));
    printf("    - Writing into buffer (cpu ptr %p, fpga ptr %p, datasize %zu)\n", src.data_, src.fpga_buffer, src.datasize_);
    cl::Event blocking_event;
    cl::Buffer *fpga_ptr = src.fpga_buffer;
    void *cpu_ptr = src.data_;
    OCL_CHECK(err, err = (*q).enqueueWriteBuffer(*fpga_ptr, CL_TRUE, 0, src.datasize_, cpu_ptr, nullptr, &blocking_event));

    printf("    - end\n");

    src.hal_ = FpgaHal::GetInstance();
    src.dev_ = Device::FPGA;
}

void FpgaHal::ToCpu(Image& src)
{
    printf("  - FPGA: ToCpu\n");
    
    if (!src.contiguous_) {
        // The copy constructor creates a new contiguous image
        src = Image(src);
    }
    printf("FpgaHal::ToCPU cp1\n");
    src.hal_ = CpuHal::GetInstance();
    src.dev_ = Device::CPU;
    printf("FpgaHal::ToCPU cp2\n");
    uint8_t* hostData = src.hal_->MemAllocate(src.datasize_);
    printf("FpgaHal::ToCPU cp3\n");
    memcpy(hostData, src.data_, src.datasize_);
    printf("FpgaHal::ToCPU cp4\n");
    free(src.data_);
    src.data_ = hostData;
    printf("FpgaHal::ToCPU cp5\n");
}

FpgaHal* FpgaHal::GetInstance()
{
#ifndef ECVL_FPGA
    ECVL_ERROR_DEVICE_UNAVAILABLE(FPGA)
#endif // ECVL_FPGA

    static FpgaHal instance; 	// Guaranteed to be destroyed.
                               // Instantiated on first use.
    return &instance;
}

void FpgaHal::CopyImage(const Image& src, Image& dst)
{
    printf("FpgaHal::CopyImage not implemented\n"); exit(1);
}

void FpgaHal::ConvertTo(const Image& src, Image& dst, DataType dtype, bool saturate)
{
    printf("FpgaHal::ConvertTo not implemented\n"); exit(1);
}


/** @brief Rearrange channels between Images of different DataTypes. */
template<DataType SDT, DataType DDT>
struct StructRearrangeImage_fpga
{
    static void _(const Image& src, Image& dst, const std::vector<int>& bindings)
    {
        using dsttype = typename TypeInfo<DDT>::basetype;
        using srctype = typename TypeInfo<SDT>::basetype;
        ConstView<SDT> vsrc(src);
        View<DDT> vdst(dst);
        auto id = vdst.Begin();

        for (size_t tmp_pos = 0; tmp_pos < dst.datasize_; tmp_pos += dst.elemsize_, ++id) {
            int x = static_cast<int>(tmp_pos);
            int src_pos = 0;
            for (int i = vsize(dst.dims_) - 1; i >= 0; i--) {
                src_pos += (x / dst.strides_[i]) * src.strides_[bindings[i]];
                x %= dst.strides_[i];
            }

            *id = static_cast<dsttype>(*reinterpret_cast<srctype*>(vsrc.data_ + src_pos));
        }
    }
};

void FpgaHal::RearrangeChannels(const Image& src, Image& dst, const std::vector<int>& bindings)
{
  printf("FPGAHal::RearrangeChannels cp1\n");
  (*q).enqueueReadBuffer(*src.fpga_buffer, CL_TRUE, 0, src.datasize_, src.data_);
  printf("FPGAHal::RearrangeChannels cp2\n");
  static constexpr Table2D<StructRearrangeImage_fpga> table;
  printf("FPGAHal::RearrangeChannels cp3\n");
  table(src.elemtype_, dst.elemtype_)(src, dst, bindings);
  printf("FPGAHal::RearrangeChannels cp4\n");

 //q.enqueueWriteBuffer(dst.fpga_buffer, CL_TRUE, 0, dst.datasize_, dst.data_);
 //    printf("FpgaHal::RearrangeChannels not implemented\n"); exit(1);
}

} // namespace ecvl
