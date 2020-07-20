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



#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "ecvl/core/datatype_matrix.h"
#include "ecvl/core/standard_errors.h"
#include <iostream>


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
	cout << "fpga haaaal Resize DIIM PRINCIPIOOO" << endl;
	OpenCVAlwaysCheck2(src);
	//cout << "fpga haaaal Resize DIIM 2222222222" << endl;
	//FromCpu(dst);
	//cout << "fpga haaaal Resize DIIM 333333333" << endl;
	//= cv::Mat::zeros(cv::Size(newdims[0], newdims[1]), CV_8UC(src_mat.channels()));
	cv::Mat src_mat = ImageToMat(src);
	cv::Mat m = cv::Mat::zeros(cv::Size(newdims[0], newdims[1]), CV_8UC(src_mat.channels()));
	ResizeDim_FPGA(src_mat, m, cv::Size(newdims[0], newdims[1]), GetOpenCVInterpolation(interp));
	dst = ecvl::MatToImage(m);

	if(dst.IsEmpty()){
		cout << "vaciaaaaaaa" << endl;
	}
	//ToCpu(dst);
	cout << "fpga haaaal Resize DIIM FINN" << endl;
}

void FpgaHal::ResizeScale(const Image& src, Image& dst, const std::vector<double>& scales, InterpolationType interp)
{
    OpenCVAlwaysCheck2(src);
	cout << "fpga haaaal Resize SCALEEE" << endl;
    int nw = lround(src.dims_[0] * scales[0]);
    int nh = lround(src.dims_[1] * scales[1]);

    cv::Mat src_mat = ImageToMat(src);
    cv::Mat m = cv::Mat::zeros(cv::Size(nw,nh), CV_8UC(src_mat.channels()));
	ResizeDim_FPGA(src_mat, m, cv::Size(nw,nh), GetOpenCVInterpolation(interp));
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
