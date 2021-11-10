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
#include "/home/jflich/git/eddl/include/eddl/hardware/fpga/xcl2.hpp"

#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ecvl/core/datatype_matrix.h"
#include "ecvl/core/standard_errors.h"
#include <iostream>

cl::CommandQueue q;
cl::Device device;
cl::Context context;
cl::Program::Binaries bins;
cl::Program program;
std::vector<cl::Device> devices;
std::string device_name;
std::string binaryFile;

#define ECVL_FPGA

namespace ecvl
{

void fpga_init(){

  devices = xcl::get_xil_devices();
  device = devices[0];
  cl_int err;

  OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
  if (err != CL_SUCCESS) printf("Error creating context 1\n");
  OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
  if (err != CL_SUCCESS) printf("Error creating command q 2\n");

  device_name = device.getInfo<CL_DEVICE_NAME>();
  binaryFile = xcl::find_binary_file(device_name,"ecvl_kernels");

  bins = xcl::import_binary_file(binaryFile);
  devices.resize(1);

  OCL_CHECK(err, program = cl::Program(context, devices, bins, NULL, &err));
  if (err != CL_SUCCESS) printf("Error creating program 3\n");

  OCL_CHECK(err, kernel_filter2d = cl::Kernel(program,"filter2d_accel", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel_filter2d \n");

  OCL_CHECK(err, kernel_warp_transform = cl::Kernel(program,"warpTransform_accel", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel_warp_transform 4\n");

  OCL_CHECK(err, kernel_resize = cl::Kernel(program,"resize_accel", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel_resize \n");

  OCL_CHECK(err, kernel_gaussian_blur = cl::Kernel(program,"gaussian_accel", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel_gaussian_blur \n");

  OCL_CHECK(err, kernel_rgb_2_gray = cl::Kernel(program,"rgb2gray_accel", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel_rgb_2_gray \n");

  OCL_CHECK(err, kernel_flip2d = cl::Kernel(program,"flipvertical_accel", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel_flip2d \n");

  OCL_CHECK(err, kernel_mirror2d = cl::Kernel(program,"mirror_accel", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel 4\n");

  OCL_CHECK(err, kernel_threshold = cl::Kernel(program,"threshold_accel", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel_threshold\n");

  OCL_CHECK(err, kernel_otsu_threshold = cl::Kernel(program,"otsuThreshold_accel", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel_otsu_threshold\n");

  cout << "END FPGA INIT" << endl;
}

FpgaHal* FpgaHal::GetInstance()
{
#ifndef ECVL_FPGA
    ECVL_ERROR_DEVICE_UNAVAILABLE(FPGA)
#endif // ECVL_FPGA

  printf("FPGA getinstance\n");

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

void FpgaHal::RearrangeChannels(const Image& src, Image& dst, const std::vector<int>& bindings)
{
    printf("FpgaHal::RearrangeChannels not implemented\n"); exit(1);
}

} // namespace ecvl
