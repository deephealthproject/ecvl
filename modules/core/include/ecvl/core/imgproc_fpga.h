#ifndef IMGPROC_FPGA_H_
#define IMGPROC_FPGA_H_

#include <opencv2/imgproc.hpp>

namespace ecvl {

void ResizeDim_FPGA(const cv::Mat& src, cv::Mat& dst, cv::Size dsize, int interp);
void Threshold_FPGA(const cv::Mat& src,  cv::Mat& dst, double thresh, double maxval);

}

#endif // IMGPROC_FPGA_H
