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


#define RGB 1
#define GRAY 0
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

static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC_T))) / (INPUT_PTR_WIDTH * NPC_T);
static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC_T))) / (OUTPUT_PTR_WIDTH * NPC_T);
void rgb2gray_accel(ap_uint<INPUT_PTR_WIDTH> *img_inp, ap_uint<OUTPUT_PTR_WIDTH> *img_out,int rows_in, int cols_in)
{
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1  depth=__XF_DEPTH_INP_0
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2  depth=__XF_DEPTH_OUT_0
#pragma HLS INTERFACE s_axilite port=img_inp               bundle=control
#pragma HLS INTERFACE s_axilite port=img_out               bundle=control
#pragma HLS INTERFACE s_axilite port=rows_in              bundle=control
#pragma HLS INTERFACE s_axilite port=cols_in              bundle=control
#pragma HLS INTERFACE s_axilite port=return                bundle=control

	
	

	xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC_T> in_mat(rows_in, cols_in);


	xf::cv::Mat<XF_8UC1,HEIGHT, WIDTH, NPC_T> out_mat(rows_in, cols_in);

#pragma HLS DATAFLOW

	xf::cv::Array2xfMat<INPUT_PTR_WIDTH,XF_8UC3,HEIGHT,WIDTH,NPC_T>(img_inp,in_mat);
	xf::cv::rgb2gray<XF_8UC3,XF_8UC1,HEIGHT,WIDTH,NPC_T> (in_mat, out_mat);
	xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH,XF_8UC1,HEIGHT,WIDTH,NPC_T>(out_mat,img_out);
}
}
