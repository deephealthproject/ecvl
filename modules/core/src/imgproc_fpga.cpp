#include "ecvl/core/imgproc_fpga.h"

#include <vector>
#include <math.h>
#include "xcl2.hpp"
#define CL_HPP_ENABLE_EXCEPTIONS
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include "/home/izcagal/eddl/src/hardware/fpga/libs/xcl2.hpp"


//#define DBG_FPGA

cl::Context context;
cl::CommandQueue q;
cl::CommandQueue com;
cl::Program program;
cl::Kernel krnl;


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

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    //cl::Context context(device);
	cl_int err;

    //cl::CommandQueue q(context, device,CL_QUEUE_PROFILING_ENABLE);
	OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel 1\n");
	OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel 2\n");

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name,"ecvl_kernels");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);

    //cl::Program program(context, devices, bins);
	OCL_CHECK(err, program = cl::Program(context, devices, bins, NULL, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel 3\n");
    //cl::Kernel krnl(program,"mirror_accel");
	OCL_CHECK(err, krnl = cl::Kernel(program,"filter2d_accel", &err));
	if (err != CL_SUCCESS) printf("Error creating kernel 4\n");

	//cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels()); // TODO check src datatype
	OCL_CHECK(err, cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels(), nullptr, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel 5\n");

	OCL_CHECK(err, cl::Buffer filterToDevice(context,CL_MEM_READ_WRITE, filter.rows * filter.cols * filter.channels() * sizeof(float), nullptr, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel filter\n");

    //cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY, dst.rows * dst.cols * dst.channels());
	OCL_CHECK(err, cl::Buffer imageFromDevice(context,CL_MEM_READ_WRITE, dst.rows * dst.cols * dst.channels(), nullptr, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel 6\n");



	OCL_CHECK(err, err = krnl.setArg(0, imageToDevice));
	if (err != CL_SUCCESS) printf("Error creating kernel 7\n");
	OCL_CHECK(err, err = krnl.setArg(1, imageFromDevice));
	if (err != CL_SUCCESS) printf("Error creating kernel 8\n");
	OCL_CHECK(err, err = krnl.setArg(2,  src.rows));
	if (err != CL_SUCCESS) printf("Error creating kernel 9\n");
	OCL_CHECK(err, err = krnl.setArg(3,  src.cols));
	if (err != CL_SUCCESS) printf("Error creating kernel 10\n");
	OCL_CHECK(err, err = krnl.setArg(4, filterToDevice));
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
    q.enqueueTask(krnl,NULL,&event_sp);
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


      std::vector<cl::Device> devices = xcl::get_xil_devices();
      cl::Device device = devices[0];
      cl::Context context(device);

      cl::CommandQueue q(context, device,CL_QUEUE_PROFILING_ENABLE);

      std::string device_name = device.getInfo<CL_DEVICE_NAME>();
      std::string binaryFile = xcl::find_binary_file(device_name,"ecvl_kernels");
      cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
      devices.resize(1);
      cl::Program program(context, devices, bins);
      cl::Kernel krnl(program,"warpTransform_accel");

      cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels()); // TODO check src datatype
      cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY, dst.rows * dst.cols * dst.channels());
      cl::Buffer imageRot(context,CL_MEM_READ_WRITE, rotMatrix.rows * rotMatrix.cols * rotMatrix.channels() * sizeof(float));

      /* Copy input vectors to memory */
      q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, src.rows * src.cols * src.channels(), src.data);
	  q.enqueueWriteBuffer(imageRot, CL_TRUE, 0, rotMatrix.rows * rotMatrix.cols * rotMatrix.channels() * sizeof(float), rotMatrix.data);

      krnl.setArg(0, imageToDevice);
      krnl.setArg(1, imageFromDevice);
      krnl.setArg(2, src.rows);
      krnl.setArg(3, src.cols);
      krnl.setArg(4, dst.rows);
      krnl.setArg(5, dst.cols);
      krnl.setArg(6, imageRot);


      // Profiling Objects
      cl_ulong start= 0;
      cl_ulong end = 0;
      double diff_prof = 0.0f;
      cl::Event event_sp;

      printf("Launching kernel: warpTransform \n");
      q.enqueueTask(krnl,NULL,&event_sp);
      clWaitForEvents(1, (const cl_event*) &event_sp);
      printf("Launched kernel: warpTransform \n");

      event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
      event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
      diff_prof = end-start;
      std::cout<<(diff_prof/1000000)<<"ms"<<std::endl;

      q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, dst.rows * dst.cols * dst.channels(), dst.data);

      q.finish();
  }


void ResizeDim_FPGA(const cv::Mat& src, cv::Mat& dst, cv::Size dsize, int interp)
{
    /* The interp parameter is ignored at the moment.
     * The xfOpenCV generates an accelerator for the Area interpolator
     * To change the accelerator interpolation strategy, its header needs to be changed,
     * and the hardware resynthesized
    */
    (void) interp;
	
	cout << "llega a imgprocFPGA" << endl;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);

    cl::CommandQueue q(context, device,CL_QUEUE_PROFILING_ENABLE);

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name,"ecvl_kernels");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl(program,"resize_accel");

    cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels()); // TODO check src datatype
    cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY, dst.rows * dst.cols * dst.channels());

    /* Copy input vectors to memory */
    q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, src.rows * src.cols * src.channels(), src.data);

    krnl.setArg(0, imageToDevice);
    krnl.setArg(1, imageFromDevice);
    krnl.setArg(2, src.rows);
    krnl.setArg(3, src.cols);
    krnl.setArg(4, dsize.height);
    krnl.setArg(5, dsize.width);

    // Profiling Objects
    cl_ulong start= 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("Launching kernel: Resize \n");
    q.enqueueTask(krnl,NULL,&event_sp);
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

	//dst = cv::Mat::zeros(src.size(), src.channels());
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);

    cl::CommandQueue q(context, device,CL_QUEUE_PROFILING_ENABLE);

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name,"ecvl_kernels");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl(program,"gaussian_accel");

    cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels()); // TODO check src datatype
    cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY, dst.rows * dst.cols * dst.channels());

    /* Copy input vectors to memory */
    q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, src.rows * src.cols * src.channels(), src.data);

    krnl.setArg(0, imageToDevice);
    krnl.setArg(1, imageFromDevice);
	krnl.setArg(2, src.rows);
    krnl.setArg(3, src.cols);
    krnl.setArg(4, sigma);


    // Profiling Objects
    cl_ulong start= 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("Launching kernel: Gaussian \n");
    q.enqueueTask(krnl,NULL,&event_sp);
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
	//dst = cv::Mat::zeros(src.size(), src.channels());
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);

    cl::CommandQueue q(context, device,CL_QUEUE_PROFILING_ENABLE);

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name,"ecvl_kernels");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl(program,"rgb2gray_accel");

    cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols * src.channels()); // TODO check src datatype
    cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY, dst.rows * dst.cols * dst.channels());




    krnl.setArg(0, imageToDevice);
    krnl.setArg(1, imageFromDevice);
	  krnl.setArg(2, src.rows);
    krnl.setArg(3, src.cols);

    /* Copy input vectors to memory */
    q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, src.rows * src.cols * src.channels(), src.data);


    // Profiling Objects
    cl_ulong start= 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("Launching kernel: rgb2gray \n");
    q.enqueueTask(krnl,NULL,&event_sp);
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

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);

    cl::CommandQueue q(context, device,CL_QUEUE_PROFILING_ENABLE);

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name,"ecvl_kernels");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
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

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);

    cl::CommandQueue q(context, device,CL_QUEUE_PROFILING_ENABLE);

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name,"ecvl_kernels");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl(program,"flipvertical_accel");

	printf("\n fuera del kernel first: %d\n", src.at<int>(0,0));
	printf("\n fuera del kernel last: %d\n", src.at<int>(899,674));

	cl::Buffer imageToDevice(context,CL_MEM_READ_WRITE, src.rows * src.cols * src.channels()); // TODO check src datatype
    cl::Buffer imageFromDevice(context,CL_MEM_READ_WRITE, dst.rows * dst.cols * dst.channels());

    /* Copy input vectors to memory */

    q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, src.rows * src.cols * src.channels(), src.data);

    krnl.setArg(0, imageToDevice);
    krnl.setArg(1, imageFromDevice);
	krnl.setArg(2, src.rows);
    krnl.setArg(3, src.cols);


    // Profiling Objects
    cl_ulong start= 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("Launching kernel: Flip2D \n");
    q.enqueueTask(krnl,NULL,&event_sp);
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

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    //cl::Context context(device);
	cl_int err;

    //cl::CommandQueue q(context, device,CL_QUEUE_PROFILING_ENABLE);
	OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel 1\n");
	OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel 2\n");

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name,"ecvl_kernels");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);

    //cl::Program program(context, devices, bins);
	OCL_CHECK(err, program = cl::Program(context, devices, bins, NULL, &err));
	if (err != CL_SUCCESS) printf("Error creating kernel 3\n");
    //cl::Kernel krnl(program,"mirror_accel");
	OCL_CHECK(err, krnl = cl::Kernel(program,"mirror_accel", &err));
	if (err != CL_SUCCESS) printf("Error creating kernel 4\n");


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

	OCL_CHECK(err, err = krnl.setArg(0, imageToDevice));
	if (err != CL_SUCCESS) printf("Error creating kernel 7\n");
	OCL_CHECK(err, err = krnl.setArg(1, imageFromDevice));
	if (err != CL_SUCCESS) printf("Error creating kernel 8\n");
	OCL_CHECK(err, err = krnl.setArg(2, src.rows));
	if (err != CL_SUCCESS) printf("Error creating kernel 9\n");
	OCL_CHECK(err, err = krnl.setArg(3, src.cols));
	if (err != CL_SUCCESS) printf("Error creating kernel 10\n");

    // Profiling Objects
    cl_ulong start= 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("Launching kernel: Mirror2D \n");
    q.enqueueTask(krnl,NULL,&event_sp);
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
  //dst = cv::Mat::zeros(src.size(), src.channels());
  std::vector<cl::Device> devices = xcl::get_xil_devices();
  cl::Device device = devices[0];
  cl::Context context(device);

  cl::CommandQueue q(context, device,CL_QUEUE_PROFILING_ENABLE);


  std::string device_name = device.getInfo<CL_DEVICE_NAME>();
  std::string binaryFile = xcl::find_binary_file(device_name,"ecvl_kernels");
  cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
  devices.resize(1);
  cl::Program program(context, devices, bins);
  cl::Kernel krnl(program,"threshold_accel");


  std::vector<cl::Memory> inBufVec, outBufVec;
  cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY,(height*width));
  cl::Buffer imageFromDevice(context, CL_MEM_WRITE_ONLY,(height*width));

  q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, (height*width), src.data);

  // Set the kernel arguments
  krnl.setArg(0, imageToDevice);
  krnl.setArg(1, imageFromDevice);
  krnl.setArg(2, thresh_uchar);
  krnl.setArg(3, maxval_uchar);
  krnl.setArg(4, height);
  krnl.setArg(5, width);

  // Profiling Objects
  cl_ulong start= 0;
  cl_ulong end = 0;
  double diff_prof = 0.0f;
  cl::Event event_sp;


  // Launch the kernel
  printf("Launching kernel: Threshold \n");
  q.enqueueTask(krnl,NULL,&event_sp);
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

  std::vector<cl::Device> devices = xcl::get_xil_devices();
  cl::Device device = devices[0];
  cl::Context context(device);

  cl::CommandQueue q(context, device,CL_QUEUE_PROFILING_ENABLE);


  std::string device_name = device.getInfo<CL_DEVICE_NAME>();
  std::string binaryFile = xcl::find_binary_file(device_name,"ecvl_kernels");
  cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
  devices.resize(1);
  cl::Program program(context, devices, bins);
  cl::Kernel krnl(program,"otsuThreshold_accel");


  std::vector<cl::Memory> inBufVec, outBufVec;
  cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY,(height*width));

  //buffer to pass the uint8t threshold to the kernel
  cl::Buffer uintToDevice(context,CL_MEM_READ_WRITE,sizeof(uint8_t));


  q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, (height*width), src.data);



  // Set the kernel arguments
  krnl.setArg(0, imageToDevice);
  krnl.setArg(1, height);
  krnl.setArg(2, width);
  krnl.setArg(3, uintToDevice);

  // Profiling Objects
  cl_ulong start= 0;
  cl_ulong end = 0;
  double diff_prof = 0.0f;
  cl::Event event_sp;


  // Launch the kernel
  printf("Launching kernel: OtsuThreshold \n");
  q.enqueueTask(krnl,NULL,&event_sp);
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
