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
#include "ap_fixed.h"
#include "assert.h"
#include "common/xf_common.h"
#include "common/xf_utility.h"
#include "imgproc/xf_remap.hpp"
#include  "hls_video.h"
#include <iostream>
#include <opencv2/core/hal/interface.h>



/* Optimization type */

#define RO 			0    // Resource Optimized (8-pixel implementation)
#define NO 			1	 // Normal Operation (1-pixel implementation)

// port widths
#define INPUT_PTR_WIDTH  256
#define OUTPUT_PTR_WIDTH 256


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
void remap_accel(ap_uint<INPUT_PTR_WIDTH> *img_inp, ap_uint<OUTPUT_PTR_WIDTH> *img_out,int rows_in, int cols_in, float *imagemap_x, float *imagemap_y)
{
	
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=imagemap_x  offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi     port=imagemap_y  offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=img_inp               bundle=control
#pragma HLS INTERFACE s_axilite port=img_out               bundle=control
#pragma HLS INTERFACE s_axilite port=rows_in              bundle=control
#pragma HLS INTERFACE s_axilite port=cols_in              bundle=control
#pragma HLS INTERFACE s_axilite port=imagemap_x              bundle=control
#pragma HLS INTERFACE s_axilite port=imagemap_y              bundle=control
#pragma HLS INTERFACE s_axilite port=return                bundle=control

	const int pROWS_INP = HEIGHT;
	const int pCOLS_INP = WIDTH;
	const int pROWS_OUT = NEWHEIGHT;
	const int pCOLS_OUT = NEWWIDTH;
	const int pNPC = NPC_T;
	
	  
/* 	xf::Mat<XF_32FC1, HEIGHT, WIDTH, XF_NPPC1> map_x;
#pragma HLS stream variable=map_x.data depth=pCOLS_INP/pNPC

	xf::Mat<XF_32FC1, HEIGHT, WIDTH, XF_NPPC1> map_y;
#pragma HLS stream variable=map_y.data depth=pCOLS_INP/pNPC

	map_x.rows = rows_in;  map_x.cols = cols_in;
	map_y.rows = rows_in;  map_y.cols = cols_in; */
	
/* 	 for(int i = 0; i<dim0;i++){
     max = A[i];
     for(int j = 0; j<dim1;j++){
       if(A[i*dim1 +j]>max){
         max = A[i*dim1 +j];
       }
     } */
	
	printf("\n dentro del kernel before: %f\n", imagemap_x[1*cols_in+500]);

	xf::Mat<XF_32FC1, HEIGHT, WIDTH, XF_NPPC1> map_x(HEIGHT, WIDTH, imagemap_x);
	xf::Mat<XF_32FC1, HEIGHT, WIDTH, XF_NPPC1> map_y(HEIGHT, WIDTH, imagemap_y);
	
	printf("\n dentro del kernel after: %f\n", map_x.read_float(1*cols_in+500));

	//printf("cols kernel: %d\n", map_x.cols);
	//printf("rows kernel: %d\n", map_x.rows);
	  

	xf::Mat<XF_8UC3, HEIGHT, WIDTH, NPC_T> in_mat;
#pragma HLS stream variable=in_mat.data depth=pCOLS_INP/pNPC

	xf::Mat<XF_8UC1,HEIGHT, WIDTH, NPC_T> out_mat;
#pragma HLS stream variable=out_mat.data depth=pCOLS_OUT/pNPC

	in_mat.rows = rows_in;  in_mat.cols = cols_in;
	out_mat.rows = rows_in;  out_mat.cols = cols_in;

#pragma HLS DATAFLOW
	
	xf::Array2xfMat<INPUT_PTR_WIDTH,XF_8UC3,HEIGHT,WIDTH,NPC_T>(img_inp,in_mat);
	xf::remap<2, XF_INTERPOLATION_NN, XF_8UC3, XF_32FC1, XF_8UC1, HEIGHT, WIDTH, XF_NPPC1, false>(in_mat, out_mat, map_x, map_y);
	xf::xfMat2Array<OUTPUT_PTR_WIDTH,XF_8UC1,HEIGHT,WIDTH,NPC_T>(out_mat,img_out);
}
}
