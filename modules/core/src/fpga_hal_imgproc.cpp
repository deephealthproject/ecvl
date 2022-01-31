/*
* ECVL - European Computer Vision Library
* Version: 0.3.4
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include "ecvl/core/fpga_hal.h"

#include <random>
#include <functional>
#include <opencv2/photo.hpp>

#if OpenCV_VERSION_MAJOR >= 4
#include <opencv2/calib3d.hpp>
#endif // #if OpenCV_VERSION_MAJOR >= 4

#include "ecvl/core/imgproc.h"
#include "ecvl/core/arithmetic.h"
#include "ecvl/core/support_opencv.h"

using namespace std;
using namespace cv;
namespace ecvl
{

cl_int err;
cl::Event event_sp;

#define RUN_KERNEL_FPGA(fn)                        \
  (*q).enqueueTask(fn, NULL, &event_sp);           \
  clWaitForEvents(1, (const cl_event*) &event_sp);

#define KERNEL_RUNTIME_PRINT \
  cl_ulong start   = 0;      \
  cl_ulong end     = 0;      \
  double diff_prof = 0.0f;   \
  event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start); \
  event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);     \
  diff_prof = end-start;                                        \
  std::cout<<(diff_prof/1000000)<<"ms"<<std::endl;


void FpgaHal::ResizeDim(const ecvl::Image& src, ecvl::Image& dst, const std::vector<int>& newdims, InterpolationType interp)
{

    int size = newdims[0] * newdims[1] * 3;

    cv::Mat m = cv::Mat::zeros(cv::Size(newdims[0], newdims[1]), CV_8UC(3));
    cl::Buffer buff = cl::Buffer(*context,CL_MEM_WRITE_ONLY, size, nullptr, &err);

    kernel_resize.setArg(0, *src.fpga_buffer);
    kernel_resize.setArg(1, buff);
    kernel_resize.setArg(2, src.dims_[1]);
    kernel_resize.setArg(3, src.dims_[0]);
    kernel_resize.setArg(4, newdims[1]);
    kernel_resize.setArg(5, newdims[0]);

    RUN_KERNEL_FPGA(kernel_resize);

    KERNEL_RUNTIME_PRINT;

    OCL_CHECK(err, err = (*q).enqueueReadBuffer(buff, CL_TRUE, 0, size, m.data));
    if (err != CL_SUCCESS) printf("Error reading buffer\n");
    (*q).finish();

    dst = ecvl::MatToImage(m);
}

void FpgaHal::ResizeScale(const Image& src, Image& dst, const std::vector<double>& scales, InterpolationType interp)
{
    int new_rows = lround(src.dims_[1] * scales[1]);
    int new_cols = lround(src.dims_[0] * scales[0]);

    cv::Mat m = cv::Mat::zeros(cv::Size(new_cols, new_rows), CV_8UC(3));
    cl::Buffer tmp_buff(*context,CL_MEM_WRITE_ONLY, m.rows * m.cols * m.channels(), nullptr, &err);

    kernel_resize.setArg(0, *src.fpga_buffer);
    kernel_resize.setArg(1, tmp_buff);
    kernel_resize.setArg(2, src.dims_[1]);
    kernel_resize.setArg(3, src.dims_[0]);
    kernel_resize.setArg(4, new_rows);
    kernel_resize.setArg(5, new_cols);

    RUN_KERNEL_FPGA(kernel_resize);

    KERNEL_RUNTIME_PRINT;

    OCL_CHECK(err, err = (*q).enqueueReadBuffer(tmp_buff, CL_TRUE, 0, new_rows * new_cols * m.channels(), m.data));
    (*q).finish();

    dst = ecvl::MatToImage(m);
}

void FpgaHal::Flip2D(const ecvl::Image& src, ecvl::Image& dst)
{
    int rows = src.dims_[0];
    int cols = src.dims_[1];

    cv::Mat m = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC(3));
    cl::Buffer tmp_buff(*context,CL_MEM_WRITE_ONLY, m.rows * m.cols * m.channels(), nullptr, &err);

    kernel_flip2d.setArg(0, *src.fpga_buffer);
    kernel_flip2d.setArg(1, tmp_buff);
    kernel_flip2d.setArg(2, src.dims_[1]);
    kernel_flip2d.setArg(3, src.dims_[0]);

    RUN_KERNEL_FPGA(kernel_flip2d);

    KERNEL_RUNTIME_PRINT;

    OCL_CHECK(err, err = (*q).enqueueReadBuffer(tmp_buff, CL_TRUE, 0, rows * cols * m.channels(), m.data));
    (*q).finish();

    dst = ecvl::MatToImage(m);
}

void FpgaHal::Mirror2D(const ecvl::Image& src, ecvl::Image& dst)
{
    int rows = src.dims_[1];
    int cols = src.dims_[0];

    cv::Mat m = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC(3));
    cl::Buffer tmp_buff(*context,CL_MEM_WRITE_ONLY, m.rows * m.cols * m.channels(), nullptr, &err);

    kernel_mirror2d.setArg(0, *src.fpga_buffer);
    kernel_mirror2d.setArg(1, tmp_buff);
    kernel_mirror2d.setArg(2, src.dims_[1]);
    kernel_mirror2d.setArg(3, src.dims_[0]);

    RUN_KERNEL_FPGA(kernel_mirror2d);
   
    KERNEL_RUNTIME_PRINT;

    (*q).enqueueReadBuffer(tmp_buff, CL_TRUE, 0, rows * cols * m.channels(), m.data);
    (*q).finish();

    dst = ecvl::MatToImage(m);
}

void FpgaHal::Rotate2D(const ecvl::Image& src, ecvl::Image& dst, double angle, const std::vector<double>& center, double scale, InterpolationType interp)
{
    printf("FpgaHal::Rotate2D not implemented\n"); exit(1);
}

void FpgaHal::RotateFullImage2D(const ecvl::Image& src, ecvl::Image& dst, double angle, double scale, InterpolationType interp)
{
    printf("FpgaHal::RotateFullImage2D not implemented\n"); exit(1);
}

void FpgaHal::ChangeColorSpace(const Image& src, Image& dst, ColorType new_type)
{
    printf("FpgaHal::ChangeColorSpace not implemented\n"); exit(1);
}

void FpgaHal::Threshold(const Image& src, Image& dst, double thresh, double maxval, ThresholdingType thresh_type)
{
    printf("FpgaHal::Threshold not implemented\n"); exit(1);
}

std::vector<double> FpgaHal::Histogram(const Image& src)
{
    printf("FpgaHal::Histogram not implemented\n"); exit(1);
}

int FpgaHal::OtsuThreshold(const Image& src)
{
    printf("FpgaHal::OtsuThreshold not implemented\n"); exit(1);
}

std::vector<int> FpgaHal::OtsuMultiThreshold(const Image& src, int n_thresholds)
{
    printf("FpgaHal::OtsuMultiThreshold not implemented\n"); exit(1);
}

void FpgaHal::MultiThreshold(const Image& src, Image& dst, const std::vector<int>& thresholds, int minval, int maxval)
{
    printf("FpgaHal::MultiThreshold not implemented\n"); exit(1);
}

void FpgaHal::Filter2D(const Image& src, Image& dst, const Image& ker, DataType type)
{
    printf("FpgaHal::Filter2D not implemented\n"); exit(1);
}

void FpgaHal::SeparableFilter2D(const Image& src, Image& dst, const vector<double>& kerX, const vector<double>& kerY, DataType type)
{
    printf("FpgaHal::SeparableFilter2D not implemented\n"); exit(1);
}

void FpgaHal::GaussianBlur(const Image& src, Image& dst, int sizeX, int sizeY, double sigmaX, double sigmaY)
{
    printf("FpgaHal::GaussianBlur not implemented\n"); exit(1);
}

void FpgaHal::AdditiveLaplaceNoise(const Image& src, Image& dst, double std_dev)
{
    printf("FpgaHal::AdditiveLaplaceNoise not implemented\n"); exit(1);
}

void FpgaHal::AdditivePoissonNoise(const Image& src, Image& dst, double lambda)
{
    printf("FpgaHal::AdditivePoissonNoise not implemented\n"); exit(1);
}

void FpgaHal::GammaContrast(const Image& src, Image& dst, double gamma)
{
    printf("FpgaHal::GammaContrast not implemented\n"); exit(1);
}

void FpgaHal::CoarseDropout(const Image& src, Image& dst, double p, double drop_size, bool per_channel)
{
    printf("FpgaHal::CoarseDropout not implemented\n"); exit(1);
}

void FpgaHal::IntegralImage(const Image& src, Image& dst, DataType dst_type)
{
    printf("FpgaHal::IntegralImage not implemented\n"); exit(1);
}

void FpgaHal::NonMaximaSuppression(const Image& src, Image& dst)
{
    printf("FpgaHal::NonMaximaSuppression not implemented\n"); exit(1);
}

vector<ecvl::Point2i> FpgaHal::GetMaxN(const Image& src, size_t n)
{
    printf("FpgaHal::GetMaxN not implemented\n"); exit(1);
}

int FpgaHal::ConnectedComponentsLabeling(const Image& src, Image& dst)
{
    printf("FpgaHal::ConnectedComponentsLabeling not implemented\n"); exit(1);
}

void FpgaHal::FindContours(const Image& src, vector<vector<ecvl::Point2i>>& contours)
{
    printf("FpgaHal::FindContours not implemented\n"); exit(1);
}

void FpgaHal::Stack(const vector<Image>& src, Image& dst)
{
    printf("FpgaHal::Stack not implemented\n"); exit(1);
}

void FpgaHal::HConcat(const vector<Image>& src, Image& dst)
{
    printf("FpgaHal::HConcat not implemented\n"); exit(1);
}

void FpgaHal::VConcat(const vector<Image>& src, Image& dst)
{
    printf("FpgaHal::VConcat not implemented\n"); exit(1);
}

void FpgaHal::Morphology(const Image& src, Image& dst, MorphType op, Image& kernel, Point2i anchor, int iterations, BorderType border_type, const int& border_value)
{
    printf("FpgaHal::Morphology not implemented\n"); exit(1);
}

void FpgaHal::Inpaint(const Image& src, Image& dst, const Image& inpaintMask, double inpaintRadius, InpaintType flag)
{
    printf("FpgaHal::Inpaint not implemented\n"); exit(1);
}

void FpgaHal::MeanStdDev(const Image& src, std::vector<double>& mean, std::vector<double>& stddev)
{
    printf("FpgaHal::MeanStdDev not implemented\n"); exit(1);
}

void FpgaHal::Transpose(const Image& src, Image& dst)
{
    printf("FpgaHal::Transpose not implemented\n"); exit(1);
}

void FpgaHal::GridDistortion(const Image& src, Image& dst, int num_steps, const std::array<float, 2>& distort_limit,
    InterpolationType interp, BorderType border_type, const int& border_value, const unsigned seed)
{
    printf("FpgaHal::GridDistortion not implemented\n"); exit(1);
}

void FpgaHal::ElasticTransform(const Image& src, Image& dst, double alpha, double sigma, InterpolationType interp,
    BorderType border_type, const int& border_value, const unsigned seed)
{
    printf("FpgaHal::ElasticTransform not implemented\n"); exit(1);
}

void FpgaHal::OpticalDistortion(const Image& src, Image& dst, const std::array<float, 2>& distort_limit, const std::array<float, 2>& shift_limit,
    InterpolationType interp, BorderType border_type, const int& border_value, const unsigned seed)
{
    printf("FpgaHal::OpticalDistortion not implemented\n"); exit(1);
}

void FpgaHal::Salt(const Image& src, Image& dst, double p, bool per_channel, const unsigned seed)
{
    printf("FpgaHal::Salt not implemented\n"); exit(1);
}

void FpgaHal::Pepper(const Image& src, Image& dst, double p, bool per_channel, const unsigned seed)
{
    printf("FpgaHal::Pepper not implemented\n"); exit(1);
}

void FpgaHal::SaltAndPepper(const Image& src, Image& dst, double p, bool per_channel, const unsigned seed)
{
    printf("FpgaHal::SaltAndPepper not implemented\n"); exit(1);
}

void FpgaHal::CentralMoments(const Image& src, Image& moments, std::vector<double> center, int order, DataType type)
{
    printf("FpgaHal::CentralMoments not implemented\n"); exit(1);
}

void FpgaHal::DrawEllipse(Image& src, ecvl::Point2i center, ecvl::Size2i axes, double angle, const ecvl::Scalar& color, int thickness)
{
    printf("FpgaHal::DrawEllipse not implemented\n"); exit(1);
}

void FpgaHal::Normalize(const Image& src, Image& dst, const double& mean, const double& std)
{
    printf("FpgaHal::Normalize not implemented\n"); exit(1);
}

void FpgaHal::Normalize(const Image& src, Image& dst, const std::vector<double>& mean, const std::vector<double>& std)
{
    printf("FpgaHal::Normalize not implemented\n"); exit(1);
}

void FpgaHal::CenterCrop(const ecvl::Image& src, ecvl::Image& dst, const std::vector<int>& size)
{
    printf("FpgaHal::CenterCrop not implemented\n"); exit(1);
}

void FpgaHal::ScaleTo(const Image& src, Image& dst, const double& new_min, const double& new_max)
{
    printf("FpgaHal::ScaleTo not implemented\n"); exit(1);
}

void FpgaHal::Pad(const Image& src, Image& dst, const vector<int>& padding, BorderType border_type, const int& border_value)
{
    printf("FpgaHal::Pad not implemented\n"); exit(1);
}

void FpgaHal::RandomCrop(const Image& src, Image& dst, const vector<int>& size, bool pad_if_needed, BorderType border_type, const int& border_value, const unsigned seed)
{
    printf("FpgaHal::RandomCrop not implemented\n"); exit(1);
}


} // namespace ecvl