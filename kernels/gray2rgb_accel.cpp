/***************************************************************************
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

***************************************************************************/

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_cvt_color.hpp"
#include "imgproc/xf_cvt_color_1.hpp"
//#include "imgproc/xf_rgb2hsv.hpp"

/* Optimization type */

#define RO 			0    // Resource Optimized (8-pixel implementation)
#define NO 			1	 // Normal Operation (1-pixel implementation)

// port widths
#define INPUT_PTR_WIDTH  512
#define OUTPUT_PTR_WIDTH 512


#define RGB 0
#define GRAY 1
/* Interpolation type*/

#define INTERPOLATION	0
// 0 - Nearest Neighbor Interpolation
// 1 - Bilinear Interpolation
// 2 - AREA Interpolation

/* Input image Dimensions */

#define WIDTH 			675	// Maximum Input image width
#define HEIGHT 			900   	// Maximum Input image height

/* Output image Dimensions */

#define NEWWIDTH 		675  // Maximum output image width
#define NEWHEIGHT 		900  // Maximum output image height

/* Interface types*/
#if RO
#define NPC_T XF_NPPC8
#else
#define NPC_T XF_NPPC1
#endif

#if RGB
#define TYPE XF_8UC3
#else
#define TYPE XF_8UC1
#endif



extern "C" {


void gray2rgb_accel(ap_uint<INPUT_PTR_WIDTH> *img_inp, ap_uint<OUTPUT_PTR_WIDTH> *img_out,int rows_in, int cols_in)
{
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1 
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2  
#pragma HLS INTERFACE s_axilite port=img_inp               
#pragma HLS INTERFACE s_axilite port=img_out            
#pragma HLS INTERFACE s_axilite port=rows_in              
#pragma HLS INTERFACE s_axilite port=cols_in            
#pragma HLS INTERFACE s_axilite port=return              

	

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC_T> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_in;
    imgInput0.cols = cols_in;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC_T> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_in;
    imgOutput0.cols = cols_in;

    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC_T>(img_inp, imgInput0);
    xf::cv::gray2rgb<XF_8UC1, XF_8UC3, HEIGHT, WIDTH, NPC_T>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC_T>(imgOutput0, img_out);

}
}
