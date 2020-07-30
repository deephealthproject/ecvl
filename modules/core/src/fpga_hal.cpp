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

#include "ecvl/core/fpga_hal.h"
#include <ecvl/core/cpu_hal.h>
#include "ecvl/core/imgproc_fpga.h"
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
#include "/home/izcagal/eddl/src/hardware/fpga/libs/xcl2.hpp"

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


// kernels
cl::Kernel kernel_resize, kernel_threshold, kernel_otsu_threshold, kernel_mirror2d, kernel_flip2d;
cl::Kernel kernel_gaussian_blur, kernel_warp_transform, kernel_rgb_2_gray, kernel_filter2d;


namespace ecvl
{
using namespace std;
FpgaHal* FpgaHal::GetInstance()
{
	//cout << "instanciaaaaa fpgaaa" << endl;
#ifndef ECVL_WITH_FPGA
    ECVL_ERROR_DEVICE_UNAVAILABLE(FPGA)
#endif // ECVL_FPGA

    static FpgaHal instance; 	// Guaranteed to be destroyed.
                               // Instantiated on first use.
    return &instance;
}


// Copy Images of different DataTypes.
template<DataType SDT, DataType DDT>
struct StructCopyImage
{
    static void _(const Image& src, Image& dst)
    {
        using dsttype = typename TypeInfo<DDT>::basetype;

        ConstView<SDT> vsrc(src);
        View<DDT> vdst(dst);
        auto is = vsrc.Begin(), es = vsrc.End();
        auto id = vdst.Begin();
        for (; is != es; ++is, ++id) {
            *id = static_cast<dsttype>(*is);
        }
    }
};
void FpgaHal::CopyImage(const Image& src, Image& dst)
{
    static constexpr Table2D<StructCopyImage> table;
    table(src.elemtype_, dst.elemtype_)(src, dst);
}

/** @brief Rearrange channels between Images of different DataTypes. */
template<DataType SDT, DataType DDT>
struct StructRearrangeImage
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
    // TODO: checks?
    static constexpr Table2D<StructRearrangeImage> table;
    table(src.elemtype_, dst.elemtype_)(src, dst, bindings);
}

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

void ReturnBuffer(cv::Mat& src, ecvl::Image& dst){
	
	cl::Buffer *imageToDevice;
	cl_int err;
	size_t size_a = src.rows * src.cols * src.channels();
	//size_t size_a = 25 * 25 * sizeof(uint8_t); -> for a vector example of 25 integers
	
	size_t size_a_in_bytes = size_a * sizeof(uint8_t);
	vector<uint8_t, aligned_allocator<uint8_t>> array(size_a, 0);
	
	//FOR LOADING IMG DATA:
/* 	if (src.isContinuous()) {
	  array.assign((uint8_t*)src.data, (uint8_t*)src.data + src.total()*src.channels());
	} else {
	  for (int i = 0; i < src.rows; ++i) {
		array.insert(array.end(), src.ptr<uint8_t>(i), src.ptr<uint8_t>(i)+src.cols*src.channels());
	  }
	} */
	
	//FOR LOADING A VECTOR EXAMPLE OF 25 INTEGERS:
	/* 	for (int i = 0; i < 25; i++) {
      for (int j = 0; j < 25; j++) {
		int ind = i*25 + j;
        array[ind] = 1;
      }
    } */
	
	//TO PRINT THE FIRST 20 MEMBERS:
/* 	for (int i = 0; i < 20; i++){
		printf("%d\n", array[i]);
	} */
	
	//WE CREATE DE CL BUFFER
	//OCL_CHECK(err, imageToDevice = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_a_in_bytes, &array[0], &err));
	//OCL_CHECK(err, imageToDevice = cl::Buffer(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels(), nullptr, &err));
	//if (err != CL_SUCCESS) printf("Error creating kernel 5\n");
	
	cl::Buffer *buffer_a = (cl::Buffer*) dst.data_;
	
	q.enqueueWriteBuffer(*buffer_a, CL_TRUE, 0, src.rows * src.cols * src.channels() * sizeof(uint8_t), src.data);

}

void FpgaHal::FromCpu(Image& src)
{
    if (!src.contiguous_) {
        // The copy constructor creates a new contiguous image
        src = Image(src);
    }
	cout << "entraaaaaaaaa FromCpu" << endl;
    uint8_t* devPtr;
    devPtr = static_cast<uint8_t *>(malloc(src.datasize_));
	cout << "malloooooc okey" << endl;
    memcpy(devPtr, src.data_, src.datasize_);
	cout << "memcpy okeeey" << endl;
/*     src.hal_->MemDeallocate(src.data_);
	cout << "MemDeallocate okeyyyy" << endl; */
    src.data_ = devPtr;
	cout << "data okeeeey " << endl;
    src.hal_ = FpgaHal::GetInstance();
    src.dev_ = Device::FPGA;
	cout << "hal y device updateeee fiiiiiin FromCpu" << endl;
}

void FpgaHal::ToCpu(Image& src)
{
    if (!src.contiguous_) {
        // The copy constructor creates a new contiguous image
        src = Image(src);
    }
	cout << "entraaaaaaaaa ToCpu" << endl;
    src.hal_ = CpuHal::GetInstance();
    src.dev_ = Device::CPU;
	cout << "hal y device updateeee" << endl;
    uint8_t* hostData = src.hal_->MemAllocate(src.datasize_);
	cout << "MemAllocate okeyyyy" << endl;
    memcpy(hostData, src.data_, src.datasize_);
	cout << "memcpy okeeey" << endl;
    free(src.data_);
    src.data_ = hostData;
	cout << "data y free okeeeey fiiin ToCpu" << endl;
}

void OpenCVAlwaysCheck2(const ecvl::Image& src)
{
    if (!(src.Width() && src.Height() && src.Channels() && vsize(src.dims_) == 3 && src.elemtype_ != DataType::int64)) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }
}

void FpgaHal::ResizeDim(const ecvl::Image& src, ecvl::Image& dst, const std::vector<int>& newdims, InterpolationType interp)
{
	cout << "fpga hal Resize DIM PRINCIPIO" << endl;
	OpenCVAlwaysCheck2(src);
	//= cv::Mat::zeros(cv::Size(newdims[0], newdims[1]), CV_8UC(src_mat.channels()));
	//cv::Mat src_mat = ImageToMat(src);
	cv::Mat m = cv::Mat::zeros(cv::Size(newdims[0], newdims[1]), CV_8UC(3));
	ResizeDim_FPGA(src, m, cv::Size(newdims[0], newdims[1]), GetOpenCVInterpolation(interp));
	dst = ecvl::MatToImage(m);

	if(dst.IsEmpty()){
		cout << "vacia" << endl;
	}
	cout << "fpga hal Resize DIM END" << endl;
}

void FpgaHal::ResizeScale(const Image& src, Image& dst, const std::vector<double>& scales, InterpolationType interp)
{
    OpenCVAlwaysCheck2(src);
	cout << "fpga haaaal Resize SCALEEE" << endl;
    int nw = lround(src.dims_[0] * scales[0]);
    int nh = lround(src.dims_[1] * scales[1]);

    cv::Mat src_mat = ImageToMat(src);
    cv::Mat m = cv::Mat::zeros(cv::Size(nw,nh), CV_8UC(src_mat.channels()));
	ResizeDim_FPGA(src, m, cv::Size(nw,nh), GetOpenCVInterpolation(interp));
    dst = ecvl::MatToImage(m);
}

void FpgaHal::Flip2D(const ecvl::Image& src, ecvl::Image& dst)
{
	OpenCVAlwaysCheck2(src);
	cv::Mat src_mat = ImageToMat(src);
	cv::Mat m = cv::Mat::zeros({ src.dims_[0], src.dims_[1] },  CV_8UC(src_mat.channels()));
		
	cv::Mat map_x = cv::Mat(cv::Size(src.dims_[0], src.dims_[1]), CV_32FC1);
	cv::Mat map_y = cv::Mat(cv::Size(src.dims_[0], src.dims_[1]), CV_32FC1);
		
	Flip2D_FPGA(src_mat, m);
	dst = ecvl::MatToImage(m);
}

void FpgaHal::Mirror2D(const ecvl::Image& src, ecvl::Image& dst)
{
	OpenCVAlwaysCheck2(src);
	cv::Mat src_mat = ImageToMat(src);
	cv::Mat m = cv::Mat::zeros({ src.dims_[0], src.dims_[1] },  CV_8UC(src_mat.channels()));
		
	cv::Mat map_x = cv::Mat(cv::Size(src.dims_[0], src.dims_[1]), CV_32FC1);
	cv::Mat map_y = cv::Mat(cv::Size(src.dims_[0], src.dims_[1]), CV_32FC1);
		
	Mirror2D_FPGA(src_mat, m);
	dst = ecvl::MatToImage(m);
}

void FpgaHal::ChangeColorSpace(const Image& src, Image& dst, ColorType new_type)
{
	OpenCVAlwaysCheck2(src);
	cv::Mat src_mat = ImageToMat(src);
	cv::Mat m = cv::Mat::zeros({ src.dims_[0], src.dims_[1] }, CV_8UC(1));
    rgb2gray_FPGA(src_mat, m);
    dst = ecvl::MatToImage(m);
}

void FpgaHal::Threshold(const Image& src, Image& dst, double thresh, double maxval, ThresholdingType thresh_type)
{
	OpenCVAlwaysCheck2(src);
    cv::Mat src_mat1 = ImageToMat(src);
    cv::Mat m = cv::Mat::zeros({ src.dims_[0], src.dims_[1] },  CV_8UC(src_mat1.channels()));
    Threshold_FPGA(src_mat1, m, thresh, maxval);
    dst = ecvl::MatToImage(m);
}

int FpgaHal::OtsuThreshold(const Image& src)
{
	OpenCVAlwaysCheck2(src);
    int threshold = 0;
    threshold = OtsuThreshold_FPGA(ImageToMat(src));
    return threshold;
}

void FpgaHal::GaussianBlur(const Image& src, Image& dst, int sizeX, int sizeY, double sigmaX, double sigmaY)
{
	
	OpenCVAlwaysCheck2(src);
	if (sizeX < 0 || (sizeX % 2 != 1)) {
		ECVL_ERROR_WRONG_PARAMS("sizeX must either be positive and odd or zero")
	}
	if (sizeY < 0 || (sizeY % 2 != 1)) {
		ECVL_ERROR_WRONG_PARAMS("sizeY must either be positive and odd or zero")
	}

	printf("sigmaX antes %f\n", sigmaX);
	printf("sigmaY antes %f\n", sigmaY);
	bool sigmaX_zero = false;
	if (sigmaX <= 0) {
		sigmaX_zero = true;
		if (sizeX == 0) {
			ECVL_ERROR_WRONG_PARAMS("sigmaX and sizeX can't be both 0")
		}
		else {
			sigmaX = 0.3 * ((sizeX - 1) * 0.5 - 1) + 0.8;
		}
	}
	
	printf("sigmaX medio %f\n", sigmaX);
	printf("sigmaY medio %f\n", sigmaY);
	if (sigmaY <= 0) {
		if (!sigmaX_zero) {
			sigmaY = sigmaX;
		}
		else if (sizeY == 0) {
			ECVL_ERROR_WRONG_PARAMS("sigmaX, sigmaY and sizeY can't be 0 at the same time")
		}
		else {
			sigmaY = 0.3 * ((sizeY - 1) * 0.5 - 1) + 0.8;
		}
	}

	if (src.channels_ != "xyc") {
		ECVL_ERROR_NOT_IMPLEMENTED
	}
	
	printf("sigmaX desp %f\n", sigmaX);
	printf("sigmaY desp %f\n", sigmaY);

	cv::Mat src_mat = ImageToMat(src);
	cv::Mat m = cv::Mat::zeros({ src.dims_[0], src.dims_[1] },  CV_8UC(src_mat.channels()));
	GaussianBlur_FPGA(src_mat, m, sigmaX);
	dst = ecvl::MatToImage(m);
}




} // namespace ecvl
