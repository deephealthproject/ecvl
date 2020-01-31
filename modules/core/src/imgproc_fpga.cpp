#include "ecvl/core/imgproc_fpga.h"

#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#include "xcl2.hpp"

namespace ecvl {

void ResizeDim_FPGA(const cv::Mat& src, cv::Mat& dst, cv::Size dsize, int interp)
{
    /* The interp parameter is ignored at the moment.
     * The xfOpenCV generates an accelerator for the Area interpolator
     * To change the accelerator interpolation strategy, its header needs to be changed,
     * and the hardware resynthesized
    */
    (void) interp;

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

    q.enqueueTask(krnl,NULL,&event_sp);
    clWaitForEvents(1, (const cl_event*) &event_sp);

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
    diff_prof = end-start;
    std::cout<<(diff_prof/1000000)<<"ms"<<std::endl;

    q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, dst.rows * dst.cols * dst.channels(), dst.data);

    q.finish();
}


void Threshold_FPGA(const cv::Mat& src, cv::Mat& dst, double thresh, double maxval){

  printf("Host program Threshold \n");
  int height = src.rows;
  int width = src.cols;
  unsigned char thresh_uchar = thresh;
  unsigned char maxval_uchar =  maxval;

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



}
