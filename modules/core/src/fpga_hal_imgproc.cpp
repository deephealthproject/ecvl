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
#include <stdio.h>      /* printf */
#include <math.h>       /* pow */
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
    int rows = src.dims_[1];
    int cols = src.dims_[0];

    cv::Mat m = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC(3));
    cl::Buffer tmp_buff(*context,CL_MEM_WRITE_ONLY, m.rows * m.cols * m.channels(), nullptr, &err);
    
    kernel_flip2d.setArg(0, *src.fpga_buffer);
    kernel_flip2d.setArg(1, tmp_buff);
    kernel_flip2d.setArg(2, rows);
    kernel_flip2d.setArg(3, cols);

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
    kernel_mirror2d.setArg(2, rows);
    kernel_mirror2d.setArg(3, cols);

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
    int rows = src.dims_[1];
    int cols = src.dims_[0];

     cv::Mat m;

    

    if (src.colortype_ == ColorType::GRAY && (new_type == ColorType::RGB || src.colortype_ == ColorType::BGR)) {
        
        m = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC(1));
        cl::Buffer tmp_buff(*context,CL_MEM_WRITE_ONLY, m.rows * m.cols * m.channels(), nullptr, &err);
        
        printf("GRAY2RGB\n"); 
        kernel_gray_2_rgb.setArg(0, *src.fpga_buffer);
        kernel_gray_2_rgb.setArg(1, tmp_buff);
        kernel_gray_2_rgb.setArg(2, rows);
        kernel_gray_2_rgb.setArg(3, cols);

        RUN_KERNEL_FPGA(kernel_gray_2_rgb);
        KERNEL_RUNTIME_PRINT;
        (*q).enqueueReadBuffer(tmp_buff, CL_TRUE, 0, rows * cols * m.channels(), m.data);

    }
    if ((src.colortype_ == ColorType::RGB || src.colortype_ == ColorType::BGR) && new_type == ColorType::GRAY){
        m = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC(3));
        cl::Buffer tmp_buff(*context,CL_MEM_WRITE_ONLY, m.rows * m.cols * m.channels(), nullptr, &err);
        
        printf("RGB2GRAY\n");
        kernel_rgb_2_gray.setArg(0, *src.fpga_buffer);
        kernel_rgb_2_gray.setArg(1, tmp_buff);
        kernel_rgb_2_gray.setArg(2, rows);
        kernel_rgb_2_gray.setArg(3, cols);

        RUN_KERNEL_FPGA(kernel_rgb_2_gray);
        KERNEL_RUNTIME_PRINT;

        (*q).enqueueReadBuffer(tmp_buff, CL_TRUE, 0, rows * cols * m.channels(), m.data);  
    }

    (*q).finish();
    dst = ecvl::MatToImage(m);
    printf("ACABA\n"); 
}

void FpgaHal::Threshold(const Image& src, Image& dst, double thresh, double maxval, ThresholdingType thresh_type)
{
    int rows = src.dims_[1];
    int cols = src.dims_[0];

    cv::Mat m = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC(1));
    cl::Buffer tmp_buff(*context,CL_MEM_WRITE_ONLY, m.rows * m.cols * m.channels(), nullptr, &err);
    
    kernel_threshold.setArg(0, *src.fpga_buffer);
    kernel_threshold.setArg(1, tmp_buff);
    kernel_threshold.setArg(2, thresh);
    kernel_threshold.setArg(3, maxval);
    kernel_threshold.setArg(4, rows);
    kernel_threshold.setArg(5, cols);

    RUN_KERNEL_FPGA(kernel_threshold);
   
    KERNEL_RUNTIME_PRINT;

    (*q).enqueueReadBuffer(tmp_buff, CL_TRUE, 0, rows * cols * m.channels(), m.data);
    (*q).finish();

    dst = ecvl::MatToImage(m);
}

std::vector<double> FpgaHal::Histogram(const Image& src)
{
    
    cl_int err;  
    std::vector<double> hist(256 * src.Channels(), 0);
    int rows = src.dims_[1];
    int cols = src.dims_[0];
    uint32_t *histogram = (uint32_t*)malloc(256 * src.Channels() * sizeof(uint32_t));
  
    
    OCL_CHECK(err,  cl::Buffer imageMapX(*context,CL_MEM_READ_WRITE, 256  * src.Channels() * sizeof(uint32_t), nullptr, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel imageMapX\n");


    // kernel_histogram.setArg(0, *src.fpga_buffer);
    // kernel_histogram.setArg(1, imageMapX);
    // kernel_histogram.setArg(2, rows);
    // kernel_histogram.setArg(3, cols);

    OCL_CHECK(err, err = kernel_histogram.setArg(0, *src.fpga_buffer));
	if (err != CL_SUCCESS) printf("Error creating args 0\n");
	OCL_CHECK(err, err = kernel_histogram.setArg(1, imageMapX));
	if (err != CL_SUCCESS) printf("Error creating args 1\n");
	OCL_CHECK(err, err = kernel_histogram.setArg(2, rows));
	if (err != CL_SUCCESS) printf("Error creating args 2\n");
	OCL_CHECK(err, err = kernel_histogram.setArg(3, cols));
	if (err != CL_SUCCESS) printf("Error creating args 3\n");

    RUN_KERNEL_FPGA(kernel_histogram);
   
    KERNEL_RUNTIME_PRINT;

    //(*q).enqueueReadBuffer(imageMapX, CL_TRUE, 0, 256 * src.Channels() * sizeof(uint32_t), histogram);
    OCL_CHECK(err, err= (*q).enqueueReadBuffer(imageMapX, CL_TRUE, 0, 256 * src.Channels() * sizeof(uint32_t), histogram));
	if (err != CL_SUCCESS) printf("Error enqueueReadBuffer\n");
    (*q).finish();

    for (int i = 0; i < hist.size(); i++)
    {
        hist[i] = (float)histogram[i];
    }
    

    return hist;

}

int FpgaHal::OtsuThreshold(const Image& src)
{
    int rows = src.dims_[1];
    int cols = src.dims_[0];

    int threshReturn;
    
    //buffer to pass the uint8t threshold to the kernel
    cl::Buffer uintToDevice(*context,CL_MEM_READ_WRITE,sizeof(int));

    kernel_threshold.setArg(0, *src.fpga_buffer);
    kernel_threshold.setArg(4, rows);
    kernel_threshold.setArg(5, cols);
    kernel_threshold.setArg(3, uintToDevice);

    RUN_KERNEL_FPGA(kernel_threshold);
   
    KERNEL_RUNTIME_PRINT;

    (*q).enqueueReadBuffer(uintToDevice, CL_TRUE, 0, sizeof(int), &threshReturn);
    (*q).finish();

    return threshReturn;
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
    int rows = src.dims_[1];
    int cols = src.dims_[0];
    cl_int err;  

    short int* filter_ptr = (short int*)malloc(3 * 3 * sizeof(short int));

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            filter_ptr[i * 3 + j] = 3640;
        }
    }
    //cl::Buffer imageMapX(*context,CL_MEM_READ_ONLY,3  * 3 * sizeof(short int), nullptr, &err);
    //(*q).enqueueWriteBuffer(imageMapX, CL_TRUE, 0, 3 *3 * sizeof(short int), filter_ptr);
    cv::Mat m = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC(3));
    //cl::Buffer tmp_buff(*context,CL_MEM_WRITE_ONLY, m.rows * m.cols * m.channels(), nullptr, &err);

    OCL_CHECK(err,  cl::Buffer imageMapX(*context,CL_MEM_READ_ONLY,3  * 3 * sizeof(short int), nullptr, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel imageMapX\n");
    OCL_CHECK(err, err= (*q).enqueueWriteBuffer(imageMapX, CL_TRUE, 0, 3 *3 * sizeof(short int), filter_ptr));
	if (err != CL_SUCCESS) printf("Error enqueueWriteBuffer\n");
	OCL_CHECK(err, cl::Buffer tmp_buff(*context,CL_MEM_WRITE_ONLY, m.rows * m.cols * m.channels(), nullptr, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel tmp_buff\n");
    /* Copy input vectors to memory */

    // kernel_filter2d.setArg(0, *src.fpga_buffer);
    // kernel_filter2d.setArg(1, tmp_buff);
    // kernel_filter2d.setArg(2, rows);
    // kernel_filter2d.setArg(3, cols);
    // kernel_filter2d.setArg(4, imageMapX);

    OCL_CHECK(err, err = kernel_filter2d.setArg(0, *src.fpga_buffer));
	if (err != CL_SUCCESS) printf("Error creating args 0\n");
	OCL_CHECK(err, err = kernel_filter2d.setArg(1, tmp_buff));
	if (err != CL_SUCCESS) printf("Error creating args 1\n");
	OCL_CHECK(err, err = kernel_filter2d.setArg(2, rows));
	if (err != CL_SUCCESS) printf("Error creating args 2\n");
	OCL_CHECK(err, err = kernel_filter2d.setArg(3, cols));
	if (err != CL_SUCCESS) printf("Error creating args 3\n");
    OCL_CHECK(err, err =  kernel_filter2d.setArg(4, imageMapX));
	if (err != CL_SUCCESS) printf("Error creating args 4\n");

    RUN_KERNEL_FPGA(kernel_filter2d);

    KERNEL_RUNTIME_PRINT;

    OCL_CHECK(err, err = (*q).enqueueReadBuffer(tmp_buff, CL_TRUE, 0, rows * cols * m.channels(), m.data));
    if (err != CL_SUCCESS) printf("Error enqueueReadBuffer\n");
    (*q).finish();


    dst = ecvl::MatToImage(m);
}

void FpgaHal::SeparableFilter2D(const Image& src, Image& dst, const vector<double>& kerX, const vector<double>& kerY, DataType type)
{
    printf("FpgaHal::SeparableFilter2D not implemented\n"); exit(1);
}

void FpgaHal::GaussianBlur(const Image& src, Image& dst, int sizeX, int sizeY, double sigmaX, double sigmaY)
{
    int rows = src.dims_[1];
    int cols = src.dims_[0];

    cv::Mat m = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC(3));
    cl::Buffer tmp_buff(*context,CL_MEM_WRITE_ONLY, m.rows * m.cols * m.channels(), nullptr, &err);
  
    kernel_gaussian_blur.setArg(0, *src.fpga_buffer);
    kernel_gaussian_blur.setArg(1, tmp_buff);
    kernel_gaussian_blur.setArg(2, rows);
    kernel_gaussian_blur.setArg(3, cols);
    kernel_gaussian_blur.setArg(4, (float)sigmaX);

    RUN_KERNEL_FPGA(kernel_gaussian_blur);
   
    KERNEL_RUNTIME_PRINT;

    (*q).enqueueReadBuffer(tmp_buff, CL_TRUE, 0, rows * cols * m.channels(), m.data);
    (*q).finish();

    dst = ecvl::MatToImage(m);

}

void FpgaHal::AdditiveLaplaceNoise(const Image& src, Image& dst, double std_dev)
{
    printf("FpgaHal::AdditiveLaplaceNoise not implemented\n"); exit(1);
}

void FpgaHal::AdditivePoissonNoise(const Image& src, Image& dst, double lambda)
{
    printf("FpgaHal::AdditivePoissonNoise not implemented\n"); exit(1);
}

void compute_gamma(float r_g, float g_g, float b_g, uchar gamma_lut[256 * 3]) {
    float gamma_inv[256] = {
        0.000000, 0.003922, 0.007843, 0.011765, 0.015686, 0.019608, 0.023529, 0.027451, 0.031373, 0.035294, 0.039216,
        0.043137, 0.047059, 0.050980, 0.054902, 0.058824, 0.062745, 0.066667, 0.070588, 0.074510, 0.078431, 0.082353,
        0.086275, 0.090196, 0.094118, 0.098039, 0.101961, 0.105882, 0.109804, 0.113725, 0.117647, 0.121569, 0.125490,
        0.129412, 0.133333, 0.137255, 0.141176, 0.145098, 0.149020, 0.152941, 0.156863, 0.160784, 0.164706, 0.168627,
        0.172549, 0.176471, 0.180392, 0.184314, 0.188235, 0.192157, 0.196078, 0.200000, 0.203922, 0.207843, 0.211765,
        0.215686, 0.219608, 0.223529, 0.227451, 0.231373, 0.235294, 0.239216, 0.243137, 0.247059, 0.250980, 0.254902,
        0.258824, 0.262745, 0.266667, 0.270588, 0.274510, 0.278431, 0.282353, 0.286275, 0.290196, 0.294118, 0.298039,
        0.301961, 0.305882, 0.309804, 0.313725, 0.317647, 0.321569, 0.325490, 0.329412, 0.333333, 0.337255, 0.341176,
        0.345098, 0.349020, 0.352941, 0.356863, 0.360784, 0.364706, 0.368627, 0.372549, 0.376471, 0.380392, 0.384314,
        0.388235, 0.392157, 0.396078, 0.400000, 0.403922, 0.407843, 0.411765, 0.415686, 0.419608, 0.423529, 0.427451,
        0.431373, 0.435294, 0.439216, 0.443137, 0.447059, 0.450980, 0.454902, 0.458824, 0.462745, 0.466667, 0.470588,
        0.474510, 0.478431, 0.482353, 0.486275, 0.490196, 0.494118, 0.498039, 0.501961, 0.505882, 0.509804, 0.513725,
        0.517647, 0.521569, 0.525490, 0.529412, 0.533333, 0.537255, 0.541176, 0.545098, 0.549020, 0.552941, 0.556863,
        0.560784, 0.564706, 0.568627, 0.572549, 0.576471, 0.580392, 0.584314, 0.588235, 0.592157, 0.596078, 0.600000,
        0.603922, 0.607843, 0.611765, 0.615686, 0.619608, 0.623529, 0.627451, 0.631373, 0.635294, 0.639216, 0.643137,
        0.647059, 0.650980, 0.654902, 0.658824, 0.662745, 0.666667, 0.670588, 0.674510, 0.678431, 0.682353, 0.686275,
        0.690196, 0.694118, 0.698039, 0.701961, 0.705882, 0.709804, 0.713725, 0.717647, 0.721569, 0.725490, 0.729412,
        0.733333, 0.737255, 0.741176, 0.745098, 0.749020, 0.752941, 0.756863, 0.760784, 0.764706, 0.768627, 0.772549,
        0.776471, 0.780392, 0.784314, 0.788235, 0.792157, 0.796078, 0.800000, 0.803922, 0.807843, 0.811765, 0.815686,
        0.819608, 0.823529, 0.827451, 0.831373, 0.835294, 0.839216, 0.843137, 0.847059, 0.850980, 0.854902, 0.858824,
        0.862745, 0.866667, 0.870588, 0.874510, 0.878431, 0.882353, 0.886275, 0.890196, 0.894118, 0.898039, 0.901961,
        0.905882, 0.909804, 0.913725, 0.917647, 0.921569, 0.925490, 0.929412, 0.933333, 0.937255, 0.941176, 0.945098,
        0.949020, 0.952941, 0.956863, 0.960784, 0.964706, 0.968627, 0.972549, 0.976471, 0.980392, 0.984314, 0.988235,
        0.992157, 0.996078, 1.000000};

    unsigned char gam_r = 0, gam_g = 0, gam_b = 0;

    for (int i = 0; i < 256; ++i) {
        // float r_inv = (float)1 / r_g;
        // float g_inv = (float)1 / g_g;
        // float b_inv = (float)1 / b_g;
        float powval_r = (float)std::pow(gamma_inv[i]/255, r_g);
        short tempgamma_r = (powval_r * 255.0);

        if (tempgamma_r > 255) {
            gam_r = 255;
        } else {
            gam_r = tempgamma_r;
        }

        float powval_g = (float)std::pow(gamma_inv[i]/255, g_g);
        short tempgamma_g = (powval_g * 255.0);

        if (tempgamma_g > 255) {
            gam_g = 255;
        } else {
            gam_g = tempgamma_g;
        }

        float powval_b = (float)std::pow(gamma_inv[i]/255, b_g);
        short tempgamma_b = (powval_b * 255.0);

        if (tempgamma_b > 255) {
            gam_b = 255;
        } else {
            gam_b = tempgamma_b;
        }
        gamma_lut[i] = gam_r;
        gamma_lut[i + 256] = gam_g;
        gamma_lut[i + 512] = gam_b;
    }
}

void FpgaHal::GammaContrast(const Image& src, Image& dst, double gamma)
{
    int rows = src.dims_[1];
    int cols = src.dims_[0];
    //unsigned char gamma_lut[256 * 3];
     cl_int err;  
 const uint8_t  gamma_lut[256 * 3] = {
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,
     2,   2,   2,   2,   2,   2,   2,   3,   3,   3,   3,   3,   3,   3,   4,   4,
     4,   4,   4,   5,   5,   5,   5,   6,   6,   6,   6,   6,   7,   7,   7,   8,
     8,   8,   8,   9,   9,   9,  10,  10,  10,  11,  11,  12,  12,  12,  13,  13,
    14,  14,  14,  15,  15,  16,  16,  17,  17,  18,  18,  19,  19,  20,  20,  21,
    22,  22,  23,  23,  24,  25,  25,  26,  27,  27,  28,  29,  29,  30,  31,  32,
    32,  33,  34,  35,  35,  36,  37,  38,  39,  40,  40,  41,  42,  43,  44,  45,
    46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  60,  61,  62,
    63,  64,  65,  67,  68,  69,  70,  72,  73,  74,  76,  77,  78,  80,  81,  82,
    84,  85,  87,  88,  90,  91,  93,  94,  96,  97,  99, 101, 102, 104, 105, 107,
   109, 111, 112, 114, 116, 118, 119, 121, 123, 125, 127, 129, 131, 132, 134, 136,
   138, 140, 142, 144, 147, 149, 151, 153, 155, 157, 159, 162, 164, 166, 168, 171,
   173, 175, 178, 180, 182, 185, 187, 190, 192, 195, 197, 200, 202, 205, 207, 210,
   213, 215, 218, 221, 223, 226, 229, 232, 235, 237, 240, 243, 246, 249, 252, 255,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,
     2,   2,   2,   2,   2,   2,   2,   3,   3,   3,   3,   3,   3,   3,   4,   4,
     4,   4,   4,   5,   5,   5,   5,   6,   6,   6,   6,   6,   7,   7,   7,   8,
     8,   8,   8,   9,   9,   9,  10,  10,  10,  11,  11,  12,  12,  12,  13,  13,
    14,  14,  14,  15,  15,  16,  16,  17,  17,  18,  18,  19,  19,  20,  20,  21,
    22,  22,  23,  23,  24,  25,  25,  26,  27,  27,  28,  29,  29,  30,  31,  32,
    32,  33,  34,  35,  35,  36,  37,  38,  39,  40,  40,  41,  42,  43,  44,  45,
    46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  60,  61,  62,
    63,  64,  65,  67,  68,  69,  70,  72,  73,  74,  76,  77,  78,  80,  81,  82,
    84,  85,  87,  88,  90,  91,  93,  94,  96,  97,  99, 101, 102, 104, 105, 107,
   109, 111, 112, 114, 116, 118, 119, 121, 123, 125, 127, 129, 131, 132, 134, 136,
   138, 140, 142, 144, 147, 149, 151, 153, 155, 157, 159, 162, 164, 166, 168, 171,
   173, 175, 178, 180, 182, 185, 187, 190, 192, 195, 197, 200, 202, 205, 207, 210,
   213, 215, 218, 221, 223, 226, 229, 232, 235, 237, 240, 243, 246, 249, 252, 255,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,
     2,   2,   2,   2,   2,   2,   2,   3,   3,   3,   3,   3,   3,   3,   4,   4,
     4,   4,   4,   5,   5,   5,   5,   6,   6,   6,   6,   6,   7,   7,   7,   8,
     8,   8,   8,   9,   9,   9,  10,  10,  10,  11,  11,  12,  12,  12,  13,  13,
    14,  14,  14,  15,  15,  16,  16,  17,  17,  18,  18,  19,  19,  20,  20,  21,
    22,  22,  23,  23,  24,  25,  25,  26,  27,  27,  28,  29,  29,  30,  31,  32,
    32,  33,  34,  35,  35,  36,  37,  38,  39,  40,  40,  41,  42,  43,  44,  45,
    46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  60,  61,  62,
    63,  64,  65,  67,  68,  69,  70,  72,  73,  74,  76,  77,  78,  80,  81,  82,
    84,  85,  87,  88,  90,  91,  93,  94,  96,  97,  99, 101, 102, 104, 105, 107,
   109, 111, 112, 114, 116, 118, 119, 121, 123, 125, 127, 129, 131, 132, 134, 136,
   138, 140, 142, 144, 147, 149, 151, 153, 155, 157, 159, 162, 164, 166, 168, 171,
   173, 175, 178, 180, 182, 185, 187, 190, 192, 195, 197, 200, 202, 205, 207, 210,
   213, 215, 218, 221, 223, 226, 229, 232, 235, 237, 240, 243, 246, 249, 252, 255,
  };
 
    cv::Mat m = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC(3));
    //cl::Buffer tmp_buff(*context,CL_MEM_WRITE_ONLY, m.rows * m.cols * m.channels(), nullptr, &err);
    //cl::Buffer buffer_inVec(*context, CL_MEM_READ_ONLY, 256 * 3 * sizeof(uint8_t));

    OCL_CHECK(err,  cl::Buffer tmp_buff(*context,CL_MEM_WRITE_ONLY, m.rows * m.cols * m.channels(), nullptr, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel tmp_buff\n");
	OCL_CHECK(err, cl::Buffer buffer_inVec(*context, CL_MEM_READ_ONLY, 256 * 3 * sizeof(uint8_t), nullptr, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel buffer_inVec\n");
    OCL_CHECK(err, err= (*q).enqueueWriteBuffer(buffer_inVec, CL_TRUE, 0, 256 * 3 * sizeof(uint8_t), gamma_lut));
	if (err != CL_SUCCESS) printf("Error enqueueWriteBuffer\n");
  
    // kernel_gamma_correction.setArg(0, *src.fpga_buffer);
    // kernel_gamma_correction.setArg(1, tmp_buff);
    // kernel_gamma_correction.setArg(2, buffer_inVec);
    // kernel_gamma_correction.setArg(3, rows);
    // kernel_gamma_correction.setArg(4, cols);
    
    OCL_CHECK(err, err = kernel_gamma_correction.setArg(0, *src.fpga_buffer));
	if (err != CL_SUCCESS) printf("Error creating args 0\n");
	OCL_CHECK(err, err = kernel_gamma_correction.setArg(1, tmp_buff));
	if (err != CL_SUCCESS) printf("Error creating args 1\n");
	OCL_CHECK(err, err = kernel_gamma_correction.setArg(2, buffer_inVec));
	if (err != CL_SUCCESS) printf("Error creating args 2\n");
	OCL_CHECK(err, err = kernel_gamma_correction.setArg(3, rows));
	if (err != CL_SUCCESS) printf("Error creating args 3\n");
    OCL_CHECK(err, err =  kernel_gamma_correction.setArg(4, cols));
	if (err != CL_SUCCESS) printf("Error creating args 4\n");

    RUN_KERNEL_FPGA(kernel_gamma_correction);
   
    KERNEL_RUNTIME_PRINT;

    //(*q).enqueueReadBuffer(tmp_buff, CL_TRUE, 0, rows * cols * m.channels(), m.data);
    OCL_CHECK(err, err= (*q).enqueueReadBuffer(tmp_buff, CL_TRUE, 0, rows * cols * m.channels(), m.data));
	if (err != CL_SUCCESS) printf("Error enqueueReadBuffer\n");
    //(*q).enqueueWriteBuffer(buffer_inVec, CL_TRUE, 0, 256 * 3 * sizeof(uint8_t), gamma_lut);
    (*q).finish();

    dst = ecvl::MatToImage(m);
}

void FpgaHal::CoarseDropout(const Image& src, Image& dst, double p, double drop_size, bool per_channel)
{
    printf("FpgaHal::CoarseDropout not implemented\n"); exit(1);
}

void FpgaHal::IntegralImage(const Image& src, Image& dst, DataType dst_type)
{
    
    int rows = src.dims_[1];
    int cols = src.dims_[0];

    cv::Mat m = cv::Mat::zeros(cv::Size(cols, rows), CV_32S);
    //cl::Buffer tmp_buff(*context,CL_MEM_WRITE_ONLY, m.rows * m.cols * 4, nullptr, &err);
    OCL_CHECK(err,  cl::Buffer tmp_buff(*context,CL_MEM_WRITE_ONLY, m.rows * m.cols * 4, nullptr, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel tmp_buff\n");
  
    // kernel_integral_image.setArg(0, *src.fpga_buffer);
    // kernel_integral_image.setArg(1, tmp_buff);
    // kernel_integral_image.setArg(2, rows);
    // kernel_integral_image.setArg(3, cols);

    OCL_CHECK(err, err = kernel_integral_image.setArg(0, *src.fpga_buffer));
	if (err != CL_SUCCESS) printf("Error creating args 0\n");
	OCL_CHECK(err, err = kernel_integral_image.setArg(1, tmp_buff));
	if (err != CL_SUCCESS) printf("Error creating args 1\n");
	OCL_CHECK(err, err = kernel_integral_image.setArg(2, rows));
	if (err != CL_SUCCESS) printf("Error creating args 2\n");
	OCL_CHECK(err, err = kernel_integral_image.setArg(3, cols));
	if (err != CL_SUCCESS) printf("Error creating args 3\n");


    RUN_KERNEL_FPGA(kernel_integral_image);
   
    KERNEL_RUNTIME_PRINT;

    //(*q).enqueueReadBuffer(tmp_buff, CL_TRUE, 0, rows * cols * 4, m.data);
    OCL_CHECK(err, err= (*q).enqueueReadBuffer(tmp_buff, CL_TRUE, 0, rows * cols * 4, m.data));
	if (err != CL_SUCCESS) printf("Error enqueueReadBuffer\n");
    (*q).finish();

    dst = ecvl::MatToImage(m);
}

void FpgaHal::NonMaximaSuppression(const Image& src, Image& dst)
{
    printf("FpgaHal::NonMaximaSuppression not implemented\n"); exit(1);
}

vector<ecvl::Point2i> FpgaHal::GetMaxN(const Image& src, size_t n)
{
    int rows = src.dims_[1];
    int cols = src.dims_[0];

    int coordsX[n];
    int coordsY[n];
    //buffer to pass the uint8t threshold to the kernel
    cl::Buffer maxlocx(*context,CL_MEM_READ_WRITE,sizeof(int)*n);
    cl::Buffer maxlocy(*context,CL_MEM_READ_WRITE,sizeof(int)*n);

    kernel_min_max.setArg(0, *src.fpga_buffer);
    kernel_min_max.setArg(1, rows);
    kernel_min_max.setArg(2, cols);
    kernel_min_max.setArg(3, maxlocx);
    kernel_min_max.setArg(4, maxlocy);
    kernel_min_max.setArg(5, n);

    RUN_KERNEL_FPGA(kernel_min_max);
   
    KERNEL_RUNTIME_PRINT;


    (*q).enqueueReadBuffer(maxlocx, CL_TRUE, 0, sizeof(int)*n, &coordsX);
    (*q).enqueueReadBuffer(maxlocy, CL_TRUE, 0, sizeof(int)*n, &coordsY);
    (*q).finish();
    

    vector<ecvl::Point2i> max_coords;
    max_coords.reserve(n);
    for (int i = 0; i < n; i++){
        printf("coords (%d,%d)\n", coordsX[i], coordsY[i]);
        max_coords[i].data()[0] = coordsX[i];
        max_coords[i].data()[1] = coordsY[i];
         printf("FpgaHal:: %d\n", max_coords[i].data()[0]);
         printf("FpgaHal:: %d\n", max_coords[i].data()[1]);
    }
    return max_coords;
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