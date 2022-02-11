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

#include "assert.h"
#include <ap_int.h>
#include <cmath>
#include <iostream>
#include <math.h>
#include <iostream>
#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_warp_transform.hpp"

//Number of fractional bits used for interpolation
#define INTER_BITS 5
#define MAX_BITS(x, y)   (((x)  > (y))? (x)  : (y))
#define INTER_TAB_SIZE (1 << INTER_BITS)
#define INTER_SCALE 1.f / INTER_TAB_SIZE

#define AB_BITS MAX_BITS(10,INTER_BITS)
#define AB_SCALE (1 << AB_BITS)
//Number of bits used to linearly interpolate
#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << INTER_REMAP_COEF_BITS)
#define ROUND_DELTA (1 << (AB_BITS - INTER_BITS - 1))


// port widths
#define INPUT_PTR_WIDTH  256
#define OUTPUT_PTR_WIDTH 256

#define INTERPOLATION	0
// 0 - Nearest Neighbor Interpolation
// 1 - Bilinear Interpolation
// 2 - AREA Interpolation


/*  set the optimisation type*/

#define NO  1  // Normal Operation
#define RO  0 // Resource Optimized

/* Input image Dimensions */

#define WIDTH 			675	// Maximum Input image width
#define HEIGHT 			900   	// Maximum Input image height

/* Output image Dimensions */

#define NEWWIDTH 		1920  // Maximum output image width
#define NEWHEIGHT 		1800  // Maximum output image height


// Number of rows of input image to be stored
#define NUM_STORE_ROWS 100

// Number of rows of input image after which output image processing must start
#define START_PROC 50

#define RGBA 0
#define GRAY 1

// transform type 0-NN 1-BILINEAR
#define INTERPOLATION 1

// transform type 0-AFFINE 1-PERSPECTIVE
#define TRANSFORM_TYPE 0
#define XF_USE_URAM false

// Set the pixel depth:
#if RGBA
#define TYPE XF_8UC3
#else
#define TYPE XF_8UC1
#endif

#define PTR_WIDTH 256

// Set the optimization type:
#define NPC1 XF_NPPC1

extern "C" {
    //using namespace std;
static constexpr int __XF_DEPTH = (NEWHEIGHT * NEWWIDTH * (XF_PIXELWIDTH(TYPE, NPC1)) / 8) / (PTR_WIDTH / 8);
void warpTransform_accel(ap_uint<INPUT_PTR_WIDTH> *img_inp, ap_uint<OUTPUT_PTR_WIDTH> *img_out, int rows_in, int cols_in, int rows_out, int cols_out, float *P_matrix)
{
  #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem0 depth=__XF_DEPTH
  #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem1 depth=__XF_DEPTH
  #pragma HLS INTERFACE m_axi     port=P_matrix  offset=slave bundle=gmem2 depth=9
  #pragma HLS INTERFACE s_axilite port=img_inp               bundle=control
  #pragma HLS INTERFACE s_axilite port=img_out               bundle=control
  #pragma HLS INTERFACE s_axilite port=P_matrix               bundle=control
  #pragma HLS INTERFACE s_axilite port=rows_in              bundle=control
  #pragma HLS INTERFACE s_axilite port=cols_in              bundle=control
  #pragma HLS INTERFACE s_axilite port=rows_out              bundle=control
  #pragma HLS INTERFACE s_axilite port=cols_out              bundle=control
  #pragma HLS INTERFACE s_axilite port=return                bundle=control


  for (int i = 0; i < 9; i++)
      printf("%f\n", P_matrix[i]);

  xf::cv::Mat<TYPE, NEWHEIGHT, NEWWIDTH, NPC1> in_mat(rows_in, cols_in);

  xf::cv::Mat<TYPE, NEWHEIGHT, NEWWIDTH, NPC1> out_mat(rows_out, cols_out);



#pragma HLS DATAFLOW
	printf("fin1\n");
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH,TYPE,NEWHEIGHT,NEWWIDTH,NPC1>(img_inp,in_mat);
    //xf::warpTransform<HEIGHT,0,0,0,TYPE,HEIGHT,WIDTH,NPC, false>(in_mat, out_mat, P_matrix);
	printf("fin2\n");
    xf::cv::warpTransform<NUM_STORE_ROWS,START_PROC,TRANSFORM_TYPE,INTERPOLATION,TYPE,NEWHEIGHT,NEWWIDTH,NPC1, XF_USE_URAM>(in_mat, out_mat, P_matrix);
	
	printf("fin3\n");
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH,TYPE,NEWHEIGHT,NEWWIDTH,NPC1>(out_mat,img_out);
	 

}
}
