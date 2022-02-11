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
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_remap.hpp"



/* Optimization type */

#define RO 			0    // Resource Optimized (8-pixel implementation)
#define NO 			1	 // Normal Operation (1-pixel implementation)


// Configure this based on the number of rows needed for the remap purpose
// e.g., If its a right to left flip two rows are enough
#define XF_WIN_ROWS 8

#define RGB 1
#define GRAY 0

#define XF_USE_URAM false

/* Interpolation type*/

// The type of interpolation, define "XF_REMAP_INTERPOLATION" as either "XF_INTERPOLATION_NN" or
// "XF_INTERPOLATION_BILINEAR"
#define XF_REMAP_INTERPOLATION XF_INTERPOLATION_BILINEAR

// Resolve interpolation type:
#if INTERPOLATION == 0
#define XF_INTERPOLATION_TYPE XF_INTERPOLATION_NN
#else
#define XF_INTERPOLATION_TYPE XF_INTERPOLATION_BILINEAR
#endif

/* Input image Dimensions */

#define WIDTH 			675	// Maximum Input image width
#define HEIGHT 			900   	// Maximum Input image height

/* Output image Dimensions */

#define NEWWIDTH 		675  // Maximum output image width
#define NEWHEIGHT 		900  // Maximum output image height

// Mat types
#define TYPE_XY XF_32FC1

#if GRAY
#define TYPE XF_8UC1
#define CHANNELS 1
#else // RGB
#define TYPE XF_8UC3
#define CHANNELS 3
#endif

// Set the optimization type:
// Only XF_NPPC1 is available for this algorithm currently
#define NPC XF_NPPC1


#define PTR_IMG_WIDTH 256
#define PTR_MAP_WIDTH 256

extern "C" {
static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE, NPC)) / 8) / (PTR_IMG_WIDTH / 8);
static constexpr int __XF_DEPTH_MAP = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE_XY, NPC)) / 8) / (4);

void remap_accel(
    ap_uint<PTR_IMG_WIDTH>* img_in, ap_uint<PTR_IMG_WIDTH>* img_out, int rows, int cols,  float* map_x, float* map_y) 
{
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=map_x         offset=slave  bundle=gmem1 depth=__XF_DEPTH_MAP
    #pragma HLS INTERFACE m_axi      port=map_y         offset=slave  bundle=gmem2 depth=__XF_DEPTH_MAP
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem3 depth=__XF_DEPTH
    #pragma HLS INTERFACE s_axilite  port=rows 	        
    #pragma HLS INTERFACE s_axilite  port=cols 	        
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC> imgInput(rows, cols);
    xf::cv::Mat<TYPE_XY, HEIGHT, WIDTH, NPC> mapX(rows, cols);
    xf::cv::Mat<TYPE_XY, HEIGHT, WIDTH, NPC> mapY(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC> imgOutput(rows, cols);

    const int HEIGHT_WIDTH_LOOPCOUNT = HEIGHT * WIDTH / XF_NPIXPERCYCLE(NPC);
    for (unsigned int i = 0; i < rows * cols; ++i) {
	// clang-format off
	#pragma HLS LOOP_TRIPCOUNT min=1 max=HEIGHT_WIDTH_LOOPCOUNT
        #pragma HLS PIPELINE II=1
        // clang-format on
        float map_x_val = map_x[i];
        float map_y_val = map_y[i];
        mapX.write_float(i, map_x_val);
        mapY.write_float(i, map_y_val);
    }

	// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_IMG_WIDTH, TYPE, HEIGHT, WIDTH, NPC>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::remap<XF_WIN_ROWS, XF_INTERPOLATION_TYPE, TYPE, TYPE_XY, TYPE, HEIGHT, WIDTH, NPC, XF_USE_URAM>(
        imgInput, imgOutput, mapX, mapY);

    // Convert _dst xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<PTR_IMG_WIDTH, TYPE, HEIGHT, WIDTH, NPC>(imgOutput, img_out);

    return;
}
}
