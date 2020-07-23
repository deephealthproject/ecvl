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

#include "ecvl/core/imgcodecs.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ecvl/core/filesystem.h"
#include "ecvl/core/support_nifti.h"
#include "ecvl/core/support_opencv.h"

#ifdef ECVL_WITH_DICOM
#include "ecvl/core/support_dcmtk.h"
#endif

#ifdef ECVL_WITH_FPGA
#include "ecvl/core/fpga_hal.h"
#include <opencv2/imgproc.hpp>
#include "xcl2.hpp"
#endif


using namespace ecvl::filesystem;

namespace ecvl
{
#ifdef ECVL_WITH_FPGA
using namespace cv;
#endif
bool ImRead(const path& filename, Image& dst, ImReadMode flags)
{
	cv::Mat src = cv::imread(filename.string(), (int)flags);
    dst = MatToImage(src);
#ifdef ECVL_WITH_DICOM
    if (dst.IsEmpty()) {
        // DICOM
        DicomRead(filename, dst);
    }
#endif
#ifdef ECVL_WITH_FPGA
	//create the context
	std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);
	
	size_t size_a = src.rows * src.cols * src.channels() * sizeof(uint8_t);
	//size_t size_a = 25 * 25 * sizeof(uint8_t); -> for a vector example of 25 integers
	vector<uint8_t, aligned_allocator<uint8_t>> array(size_a, 0);
	
	//FOR LOADING IMG DATA:
	if (src.isContinuous()) {
	  array.assign((uint8_t*)src.data, (uint8_t*)src.data + src.total()*src.channels());
	} else {
	  for (int i = 0; i < src.rows; ++i) {
		array.insert(array.end(), src.ptr<uint8_t>(i), src.ptr<uint8_t>(i)+src.cols*src.channels());
	  }
	}
	
	//FOR LOADING A VECTOR EXAMPLE OF 25 INTEGERS:
/* 	for (int i = 0; i < 25; i++) {
      for (int j = 0; j < 25; j++) {
		int ind = i*25 + j;
        array[ind] = 1;
      }
    } */
	
	//TO PRINT THE FIRST 20 MEMBERS:
	for (int i = 0; i < 20; i++){
		printf("%d\n", array[i]);
	}
	
	//WE CREATE DE CL BUFFER
	dst.data_ = (uint8_t *) new cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_a, &array[0], nullptr);
	dst.dev_ = ecvl::Device::FPGA;
	dst.hal_ = FpgaHal::GetInstance();
#endif
    if (dst.IsEmpty()) {
        // NIFTI
        NiftiRead(filename, dst);
    }
    return !dst.IsEmpty();
}

bool ImReadMulti(const path& filename, Image& dst)
{
    std::vector<cv::Mat> v;
    cv::imreadmulti(filename.string(), v);
    dst = MatVecToImage(v);

    return !dst.IsEmpty();
}

bool ImWrite(const path& filename, const Image& src)
{
    return cv::imwrite(filename.string(), ImageToMat(src));
}
} // namespace ecvl