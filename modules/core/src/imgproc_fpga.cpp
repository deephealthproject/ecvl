#include "ecvl/core/imgproc_fpga.h"

#include <vector>

#include <CL/cl.h>
#include "xcl2.hpp"
//#include "ap_int.h"

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
    std::string binaryFile = xcl::find_binary_file(device_name,"krnl_resize");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl(program,"resize_accel");

    cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, src.rows * src.cols); // TODO check src datatype
    cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY, dsize.area());

    /* Copy input vectors to memory */
    //q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, src.rows * src.cols, (ap_uint<INPUT_PTR_WIDTH>*)src.data);
    q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, src.rows * src.cols, src.data);

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

    q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, dsize.area(), dst.data);

    q.finish();
}

}
