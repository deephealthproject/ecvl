#include "ecvl/core/imgproc_fpga.h"

#include <vector>
#include <math.h>
#include "xcl2.hpp"
#define CL_HPP_ENABLE_EXCEPTIONS
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include "ecvl/core/image.h"
#include "/home/izcagal/eddl/src/hardware/fpga/libs/xcl2.hpp"

//#define DBG_FPGA


extern cl::CommandQueue q;
extern cl::Device device;
extern cl::Context context;
extern cl::Program::Binaries bins;
extern cl::Program program;
extern cl::Buffer imageToDevice;
extern std::vector<cl::Device> devices;
extern std::string device_name;
extern std::string binaryFile;

// kernels
extern cl::Kernel kernel_resize, kernel_threshold, kernel_otsu_threshold, kernel_mirror2d, kernel_flip2d;
extern cl::Kernel kernel_gaussian_blur, kernel_warp_transform, kernel_rgb_2_gray, kernel_filter2d, krnl;


namespace ecvl {
using namespace cv;
using namespace std;

void Filter2d_FPGA(const cv::Mat& src, cv::Mat& dst, cv::Mat& filter, int width, int height)
{
    /* The interp parameter is ignored at the moment.
     * The xfOpenCV generates an accelerator for the Area interpolator
     * To change the accelerator interpolation strategy, its header needs to be changed,
     * and the hardware resynthesized
    */

	//cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels()); // TODO check src datatype
	cl_int err;
	OCL_CHECK(err, cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels(), nullptr, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel 5\n");

	OCL_CHECK(err, cl::Buffer filterToDevice(context,CL_MEM_READ_WRITE, filter.rows * filter.cols * filter.channels() * sizeof(float), nullptr, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel filter\n");

    //cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY, dst.rows * dst.cols * dst.channels());
	OCL_CHECK(err, cl::Buffer imageFromDevice(context,CL_MEM_READ_WRITE, dst.rows * dst.cols * dst.channels(), nullptr, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel 6\n");



	OCL_CHECK(err, err = kernel_filter2d.setArg(0, imageToDevice));
	if (err != CL_SUCCESS) printf("Error creating kernel 7\n");
	OCL_CHECK(err, err = kernel_filter2d.setArg(1, imageFromDevice));
	if (err != CL_SUCCESS) printf("Error creating kernel 8\n");
	OCL_CHECK(err, err = kernel_filter2d.setArg(2,  src.rows));
	if (err != CL_SUCCESS) printf("Error creating kernel 9\n");
	OCL_CHECK(err, err = kernel_filter2d.setArg(3,  src.cols));
	if (err != CL_SUCCESS) printf("Error creating kernel 10\n");
	OCL_CHECK(err, err = kernel_filter2d.setArg(4, filterToDevice));
	if (err != CL_SUCCESS) printf("Error creating kernel 11\n");


    /* Copy input vectors to memory */
	OCL_CHECK(err, err= q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, src.rows * src.cols * src.channels(), src.data));
	if (err != CL_SUCCESS) printf("Error encolando\n");

	OCL_CHECK(err, err= q.enqueueWriteBuffer(filterToDevice, CL_TRUE, 0, filter.rows * filter.cols * filter.channels() * sizeof(float), filter.data));
	if (err != CL_SUCCESS) printf("Error encolando filter\n");


    // Profiling Objects
    cl_ulong start= 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("Launching kernel: filter2D \n");
    q.enqueueTask(kernel_filter2d,NULL,&event_sp);
    clWaitForEvents(1, (const cl_event*) &event_sp);
    printf("Launched kernel: filter2D \n");

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
    diff_prof = end-start;
    std::cout<<(diff_prof/1000000)<<"ms"<<std::endl;

	 printf("AQUIIIIIIIIIIIIIIIIII\n");
	OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, dst.rows * dst.cols * dst.channels(), dst.data));
	if (err != CL_SUCCESS) printf("Error retornando\n");
	printf("AQUIIIIIIIIIIIIIIIIII22222222222222\n");
    q.finish();
}

  void warpTransform_FPGA(const cv::Mat& src, cv::Mat& dst, cv::Mat& rotMatrix)
  {
      /* The interp parameter is ignored at the moment.
       * The xfOpenCV generates an accelerator for the Area interpolator
       * To change the accelerator interpolation strategy, its header needs to be changed,
       * and the hardware resynthesized
      */


      cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels()); // TODO check src datatype
      cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY, dst.rows * dst.cols * dst.channels());
      cl::Buffer imageRot(context,CL_MEM_READ_WRITE, rotMatrix.rows * rotMatrix.cols * rotMatrix.channels() * sizeof(float));

      /* Copy input vectors to memory */
      q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, src.rows * src.cols * src.channels(), src.data);
	  q.enqueueWriteBuffer(imageRot, CL_TRUE, 0, rotMatrix.rows * rotMatrix.cols * rotMatrix.channels() * sizeof(float), rotMatrix.data);

      kernel_warp_transform.setArg(0, imageToDevice);
      kernel_warp_transform.setArg(1, imageFromDevice);
      kernel_warp_transform.setArg(2, src.rows);
      kernel_warp_transform.setArg(3, src.cols);
      kernel_warp_transform.setArg(4, dst.rows);
      kernel_warp_transform.setArg(5, dst.cols);
      kernel_warp_transform.setArg(6, imageRot);


      // Profiling Objects
      cl_ulong start= 0;
      cl_ulong end = 0;
      double diff_prof = 0.0f;
      cl::Event event_sp;

      printf("Launching kernel: warpTransform \n");
      q.enqueueTask(kernel_warp_transform,NULL,&event_sp);
      clWaitForEvents(1, (const cl_event*) &event_sp);
      printf("Launched kernel: warpTransform \n");

      event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
      event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
      diff_prof = end-start;
      std::cout<<(diff_prof/1000000)<<"ms"<<std::endl;

      q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, dst.rows * dst.cols * dst.channels(), dst.data);

      q.finish();
  }


void ResizeDim_FPGA(const ecvl::Image& src, cv::Mat& dst, cv::Size dsize, int interp)
{
    /* The interp parameter is ignored at the moment.
     * The xfOpenCV generates an accelerator for the Area interpolator
     * To change the accelerator interpolation strategy, its header needs to be changed,
     * and the hardware resynthesized
    */
    (void) interp;
	
	cout << "llega a imgprocFPGA" << endl;

    //cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels()); // TODO check src datatype
    cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY, dst.rows * dst.cols * dst.channels());
	
	//Cl buffer with the original imgdata
	cl::Buffer *buffer_a = (cl::Buffer*) src.data_;
	
	
    /* Copy input vectors to memory -> Now we create a cl buffer into the image, it is not necessary */
    //q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, src.rows * src.cols * src.channels(), src.data);
	//cl::Event  event2;
	//q.enqueueMigrateMemObjects({*src.buffer_},0);
	
	
    kernel_resize.setArg(0, *buffer_a);
    kernel_resize.setArg(1, imageFromDevice);
    kernel_resize.setArg(2, src.dims_[1]);
    kernel_resize.setArg(3, src.dims_[0]);
    kernel_resize.setArg(4, dsize.height);
    kernel_resize.setArg(5, dsize.width);

    // Profiling Objects
    cl_ulong start= 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("Launching kernel: Resize \n");
    q.enqueueTask(kernel_resize,NULL,&event_sp);
    clWaitForEvents(1, (const cl_event*) &event_sp);
    printf("Launched kernel: Resize \n");

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
    diff_prof = end-start;
    std::cout<<(diff_prof/1000000)<<"ms"<<std::endl;

    q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, dst.rows * dst.cols * dst.channels(), dst.data);

    q.finish();
}



void GaussianBlur_FPGA(const cv::Mat& src, cv::Mat& dst, float sigma)
{
    /* The interp parameter is ignored at the moment.
     * The xfOpenCV generates an accelerator for the Area interpolator
     * To change the accelerator interpolation strategy, its header needs to be changed,
     * and the hardware resynthesized
    */

    cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels()); // TODO check src datatype
    cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY, dst.rows * dst.cols * dst.channels());

    /* Copy input vectors to memory */
    q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, src.rows * src.cols * src.channels(), src.data);

    kernel_gaussian_blur.setArg(0, imageToDevice);
    kernel_gaussian_blur.setArg(1, imageFromDevice);
	kernel_gaussian_blur.setArg(2, src.rows);
    kernel_gaussian_blur.setArg(3, src.cols);
    kernel_gaussian_blur.setArg(4, sigma);


    // Profiling Objects
    cl_ulong start= 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("Launching kernel: Gaussian \n");
    q.enqueueTask(kernel_gaussian_blur,NULL,&event_sp);
    clWaitForEvents(1, (const cl_event*) &event_sp);
    printf("Launched kernel: Gaussian \n");

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
    diff_prof = end-start;
    std::cout<<(diff_prof/1000000)<<"ms"<<std::endl;

    q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, dst.rows * dst.cols * dst.channels(), dst.data);

    q.finish();
}

void rgb2gray_FPGA(const cv::Mat& src, cv::Mat& dst)
{
    /* The interp parameter is ignored at the moment.
     * The xfOpenCV generates an accelerator for the Area interpolator
     * To change the accelerator interpolation strategy, its header needs to be changed,
     * and the hardware resynthesized
    */

    cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels()); // TODO check src datatype
    cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY, dst.rows * dst.cols * dst.channels());

    kernel_rgb_2_gray.setArg(0, imageToDevice);
    kernel_rgb_2_gray.setArg(1, imageFromDevice);
	kernel_rgb_2_gray.setArg(2, src.rows);
    kernel_rgb_2_gray.setArg(3, src.cols);

    /* Copy input vectors to memory */
    q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, src.rows * src.cols * src.channels(), src.data);


    // Profiling Objects
    cl_ulong start= 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("Launching kernel: rgb2gray \n");
    q.enqueueTask(kernel_rgb_2_gray,NULL,&event_sp);
    clWaitForEvents(1, (const cl_event*) &event_sp);
    printf("Launched kernel: rgb2gray \n");

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
    diff_prof = end-start;
    std::cout<<(diff_prof/1000000)<<"ms"<<std::endl;

    q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, dst.rows * dst.cols * dst.channels(), dst.data);

    q.finish();
}

//Defines the type of remap
void update_map( int &ind, Mat &map_x, Mat &map_y )
{
    for( int i = 0; i < map_x.rows; i++ )
    {
        for( int j = 0; j < map_x.cols; j++ )
        {
            switch( ind )
            {
            case 0:
                if( j > map_x.cols*0.25 && j < map_x.cols*0.75 && i > map_x.rows*0.25 && i < map_x.rows*0.75 )
                {
                    map_x.at<float>(i, j) = 2*( j - map_x.cols*0.25f ) + 0.5f;
                    map_y.at<float>(i, j) = 2*( i - map_x.rows*0.25f ) + 0.5f;
                }
                else
                {
                    map_x.at<float>(i, j) = 0;
                    map_y.at<float>(i, j) = 0;
                }
                break;
            case 1:
                map_x.at<float>(i, j) = (float)j;
                map_y.at<float>(i, j) = (float)(map_x.rows - i);
                break;
            case 2:
                map_x.at<float>(i, j) = (float)(map_x.cols - j);
                map_y.at<float>(i, j) = (float)i;
                break;
            case 3:
                map_x.at<float>(i, j) = (float)(map_x.cols - j);
                map_y.at<float>(i, j) = (float)(map_x.rows - i);
                break;
            default:
                break;
            } // end of switch
        }
    }
}

void remap_FPGA(const cv::Mat& src, cv::Mat& dst, cv::Mat& map_x, cv::Mat& map_y, int type)
{
    /* The interp parameter is ignored at the moment.
     * The xfOpenCV generates an accelerator for the Area interpolator
     * To change the accelerator interpolation strategy, its header needs to be changed,
     * and the hardware resynthesized
    */
    cl::Kernel krnl(program,"remap_accel");

	/*
	*  Prepating the kind of remap
	*/

	update_map(type, map_x, map_y);

/* 	xf::Mat<XF_32FC1, HEIGHT, WIDTH, XF_NPPC1> map_x(HEIGHT, WIDTH);
	xf::Mat<XF_32FC1, HEIGHT, WIDTH, XF_NPPC1> map_y(HEIGHT, WIDTH);

	map_x.copyTo(imageMapX->data);
	map_y.copyTo(imageMapY->data); */

	 //cout << map_x.at<double>(0,250);
	printf("\n fuera del kernel after: %f\n", map_x.at<float>(1,500));



    cl::Buffer imageMapX(context,CL_MEM_READ_ONLY, map_x.rows * map_x.cols * map_x.channels() * sizeof(float)); // TODO check src datatype
	cl::Buffer imageMapY(context,CL_MEM_READ_ONLY, map_y.rows * map_y.cols * map_y.channels() * sizeof(float)); // TODO check src datatype
	cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels()); // TODO check src datatype
    cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY, dst.rows * dst.cols * dst.channels());

    /* Copy input vectors to memory */
    q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, src.rows * src.cols * src.channels(), src.data);
	q.enqueueWriteBuffer(imageMapX, CL_TRUE, 0, map_x.rows * map_x.cols * map_x.channels() * sizeof(float), map_x.data);
	q.enqueueWriteBuffer(imageMapY, CL_TRUE, 0, map_y.rows * map_y.cols * map_y.channels() * sizeof(float), map_y.data);

    krnl.setArg(0, imageToDevice);
    krnl.setArg(1, imageFromDevice);
	krnl.setArg(2, src.rows);
    krnl.setArg(3, src.cols);
	krnl.setArg(4, imageMapX);
    krnl.setArg(5, imageMapY);


    // Profiling Objects
    cl_ulong start= 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("Launching kernel: remap \n");
    q.enqueueTask(krnl,NULL,&event_sp);
    clWaitForEvents(1, (const cl_event*) &event_sp);
    printf("Launched kernel: remap \n");

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
    diff_prof = end-start;
    std::cout<<(diff_prof/1000000)<<"ms"<<std::endl;

    q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, dst.rows * dst.cols * dst.channels(), dst.data);

    q.finish();
}

void Flip2D_FPGA(const cv::Mat& src, cv::Mat& dst)
{
    /* The interp parameter is ignored at the moment.
     * The xfOpenCV generates an accelerator for the Area interpolator
     * To change the accelerator interpolation strategy, its header needs to be changed,
     * and the hardware resynthesized
    */

	printf("\n fuera del kernel first: %d\n", src.at<int>(0,0));
	printf("\n fuera del kernel last: %d\n", src.at<int>(899,674));

	cl::Buffer imageToDevice(context,CL_MEM_READ_WRITE, src.rows * src.cols * src.channels()); // TODO check src datatype
    cl::Buffer imageFromDevice(context,CL_MEM_READ_WRITE, dst.rows * dst.cols * dst.channels());

    /* Copy input vectors to memory */

    q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, src.rows * src.cols * src.channels(), src.data);

    kernel_flip2d.setArg(0, imageToDevice);
    kernel_flip2d.setArg(1, imageFromDevice);
	kernel_flip2d.setArg(2, src.rows);
    kernel_flip2d.setArg(3, src.cols);


    // Profiling Objects
    cl_ulong start= 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("Launching kernel: Flip2D \n");
    q.enqueueTask(kernel_flip2d,NULL,&event_sp);
    clWaitForEvents(1, (const cl_event*) &event_sp);
    printf("Launched kernel: Flip2D \n");

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
    diff_prof = end-start;
    std::cout<<(diff_prof/1000000)<<"ms"<<std::endl;

    q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, dst.rows * dst.cols * dst.channels(), dst.data);

    q.finish();
}

void Mirror2D_FPGA(const cv::Mat& src, cv::Mat& dst)
{
    /* The interp parameter is ignored at the moment.
     * The xfOpenCV generates an accelerator for the Area interpolator
     * To change the accelerator interpolation strategy, its header needs to be changed,
     * and the hardware resynthesized
    */


	cl_int err;
	printf("\n fuera del kernel first: %d\n", src.at<int>(0,0));
	printf("\n fuera del kernel last: %d\n", src.at<int>(899,674));

	//cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels()); // TODO check src datatype
	OCL_CHECK(err, cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels(), nullptr, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel 5\n");
    //cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY, dst.rows * dst.cols * dst.channels());
	OCL_CHECK(err, cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY, dst.rows * dst.cols * dst.channels(), nullptr, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel 6\n");
    /* Copy input vectors to memory */
	OCL_CHECK(err, err= q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, src.rows * src.cols * src.channels(), src.data));
	if (err != CL_SUCCESS) printf("Error encolando\n");

	OCL_CHECK(err, err = kernel_mirror2d.setArg(0, imageToDevice));
	if (err != CL_SUCCESS) printf("Error creating kernel 7\n");
	OCL_CHECK(err, err = kernel_mirror2d.setArg(1, imageFromDevice));
	if (err != CL_SUCCESS) printf("Error creating kernel 8\n");
	OCL_CHECK(err, err = kernel_mirror2d.setArg(2, src.rows));
	if (err != CL_SUCCESS) printf("Error creating kernel 9\n");
	OCL_CHECK(err, err = kernel_mirror2d.setArg(3, src.cols));
	if (err != CL_SUCCESS) printf("Error creating kernel 10\n");

    // Profiling Objects
    cl_ulong start= 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("Launching kernel: Mirror2D \n");
    q.enqueueTask(kernel_mirror2d,NULL,&event_sp);
    clWaitForEvents(1, (const cl_event*) &event_sp);
    printf("Launched kernel: Mirror2D \n");

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
    diff_prof = end-start;
    std::cout<<(diff_prof/1000000)<<"ms"<<std::endl;

	OCL_CHECK(err, err = q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, dst.rows * dst.cols * dst.channels(), dst.data));
	if (err != CL_SUCCESS) printf("Error retornando\n");

    q.finish();
}

void Threshold_FPGA(const cv::Mat& src, cv::Mat& dst, double thresh, double maxval){

  printf("Host program Threshold \n");
  int height = src.rows;
  int width = src.cols;
  unsigned char thresh_uchar = thresh;
  unsigned char maxval_uchar =  maxval;

  cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY,(height*width));
  cl::Buffer imageFromDevice(context, CL_MEM_WRITE_ONLY,(height*width));

  q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, (height*width), src.data);

  // Set the kernel arguments
  kernel_threshold.setArg(0, imageToDevice);
  kernel_threshold.setArg(1, imageFromDevice);
  kernel_threshold.setArg(2, thresh_uchar);
  kernel_threshold.setArg(3, maxval_uchar);
  kernel_threshold.setArg(4, height);
  kernel_threshold.setArg(5, width);

  // Profiling Objects
  cl_ulong start= 0;
  cl_ulong end = 0;
  double diff_prof = 0.0f;
  cl::Event event_sp;


  // Launch the kernel
  printf("Launching kernel: Threshold \n");
  q.enqueueTask(kernel_threshold,NULL,&event_sp);
  clWaitForEvents(1, (const cl_event*) &event_sp);
  printf("Launched kernel: Threshold \n");


  event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
  event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
  diff_prof = end-start;
  std::cout<<(diff_prof/1000000)<<"ms"<<std::endl;


  //Copying Device result data to Host memory
  q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, (height*width), dst.data);
  printf("Finish kernel: Threshold \n");
  q.finish();

}

uint8_t OtsuThreshold_FPGA(const cv::Mat& src){

  printf("Host program OtsuThreshold \n");
  int height = src.rows;
  int width = src.cols;
	
  uint8_t threshReturn;


  cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY,(height*width));

  //buffer to pass the uint8t threshold to the kernel
  cl::Buffer uintToDevice(context,CL_MEM_READ_WRITE,sizeof(uint8_t));
  q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, (height*width), src.data);

  // Set the kernel arguments
  kernel_otsu_threshold.setArg(0, imageToDevice);
  kernel_otsu_threshold.setArg(1, height);
  kernel_otsu_threshold.setArg(2, width);
  kernel_otsu_threshold.setArg(3, uintToDevice);

  // Profiling Objects
  cl_ulong start= 0;
  cl_ulong end = 0;
  double diff_prof = 0.0f;
  cl::Event event_sp;


  // Launch the kernel
  printf("Launching kernel: OtsuThreshold \n");
  q.enqueueTask(kernel_otsu_threshold,NULL,&event_sp);
  clWaitForEvents(1, (const cl_event*) &event_sp);
  printf("Launched kernel: OtsuThreshold \n");


  event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
  event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
  diff_prof = end-start;
  std::cout<<(diff_prof/1000000)<<"ms"<<std::endl;


  //Copying Device result data to Host memory
  //q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, (height*width), dst.data);

  //queue to get de data from the buffer when the kernel has finished
  q.enqueueReadBuffer(uintToDevice, CL_TRUE, 0, sizeof(uint8_t), &threshReturn);
  printf("Finish kernel: OtsuThreshold\n");
  q.finish();


  return threshReturn;


}



}
