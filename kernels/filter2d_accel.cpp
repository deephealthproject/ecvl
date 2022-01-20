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
#include <stdint.h>
#include <stdio.h>
#include "ap_int.h"
#include "common/xf_common.h"
#include <iostream>
#include  "hls_video.h"



/* Optimization type */

#define RO 			0    // Resource Optimized (8-pixel implementation)
#define NO 			1	 // Normal Operation (1-pixel implementation)

// port widths
#define INPUT_PTR_WIDTH  32
#define OUTPUT_PTR_WIDTH 32

//Filter size. Filter size of 3 (XF_FILTER_3X3), 5 (XF_FILTER_5X5) and 7 (XF_FILTER_7X7) aresupported

#define FILTER_SIZE 5

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

#define NEWWIDTH 		1920  // Maximum output image width
#define NEWHEIGHT 		1800  // Maximum output image height

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
	using namespace std;
void filter2d_accel(ap_uint<INPUT_PTR_WIDTH> *img_inp, ap_uint<OUTPUT_PTR_WIDTH> *img_out,int rows_in, int cols_in, float *filter)
{
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=filter  offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=img_inp               bundle=control
#pragma HLS INTERFACE s_axilite port=img_out               bundle=control
#pragma HLS INTERFACE s_axilite port=rows_in              bundle=control
#pragma HLS INTERFACE s_axilite port=cols_in              bundle=control
#pragma HLS INTERFACE s_axilite port=filter              bundle=control
#pragma HLS INTERFACE s_axilite port=return                bundle=control

	const int pROWS_INP = HEIGHT;
	const int pCOLS_INP = WIDTH;
	const int pROWS_OUT = NEWHEIGHT;
	const int pCOLS_OUT = NEWWIDTH;
	const int pNPC = NPC_T;

/* 	xf::Mat<TYPE, HEIGHT, WIDTH, NPC_T> in_mat;
#pragma HLS stream variable=in_mat.data depth=pCOLS_INP/pNPC

	xf::Mat<TYPE, HEIGHT, WIDTH, NPC_T> out_mat;
#pragma HLS stream variable=out_mat.data depth=pCOLS_OUT/pNPC

	in_mat.rows = rows_in;  in_mat.cols = cols_in;
	out_mat.rows = rows_in;  out_mat.cols = cols_in; */

	//uint32_t image_in[HEIGHT*WIDTH] = {};

#pragma HLS DATAFLOW
	printf("holaaaaaaaaaaaaaaaaaaaa\n");


 	hls::Mat<HEIGHT,WIDTH,HLS_8UC3> src(rows_in, cols_in);
	#pragma HLS stream variable=src depth=pCOLS_INP/pNPC
	hls::Mat<HEIGHT,WIDTH,HLS_8UC3> dst(rows_in, cols_in);
	#pragma HLS stream variable=dst depth=pCOLS_OUT/pNPC
/* 	src.rows = rows_in;  src.cols = cols_in;
	dst.rows = rows_in;  dst.cols = cols_in; */

	printf("cols in: %d\n", cols_in);
	printf("rows in: %d\n", rows_in);

	printf("holaaaaaaaaaaaaaaaaaaaa2\n");


/* 	for (int i = 0; i < rows_in*cols_in; i++) {
		#pragma HLS loop_flatten off
		#pragma HLS pipeline II=1
			image_in[i]=img_inp[i];
	} */

/* 	int fb_BitWidth = Type_BitWidth<ap_uint<INPUT_PTR_WIDTH>>::Value;
    int depth = HLS_TBITDEPTH(HLS_8UC3);
    int ch = HLS_MAT_CN(HLS_8UC3);
	printf("%d >= %d * %d\n", fb_BitWidth, ch, depth); */
	//::Array2Mat<WIDTH*3>(img_inp,src);


	hls::Array2Mat<WIDTH,ap_uint<INPUT_PTR_WIDTH>,HEIGHT,WIDTH,HLS_8UC3>(img_inp,src);


	printf("holaaaaaaaaaaaaaaaaaaaa3\n");
	hls::Window<3,3,float> kernel;

	for (int i=0;i<3;i++){
      for (int j=0;j<3;j++){
         kernel.val[i][j]=filter[i*3+j];
      }
	}

	for (int i=0;i<3;i++){
      for (int j=0;j<3;j++){
         printf("%f\n", kernel.val[i][j]);
      }
	}


	hls::Point_<int> anchor = hls::Point_<int>(-1,-1);
	printf("holaaaaaaaaaaaaaaaaaaaa4\n");

	std::cout << "cols: " << src.cols << std::endl;

	std::cout << "rows: " << src.rows << std::endl;

	hls::Filter2D(src,dst,kernel,anchor);
	printf("holaaaaaaaaaaaaaaaaaaaa5\n");


	hls::Mat2Array<WIDTH,ap_uint<OUTPUT_PTR_WIDTH>,HEIGHT,WIDTH,HLS_8UC3>(dst,img_out);
	//hlsMat2cvMat<HEIGHT,WIDTH,HLS_8UC3>(dst,img_out);
	// uint32_t res = img_out[0].to_int();
	// printf("%d\n", res);
	std::cout << "return: " << img_out[208000] << std::endl;
	std::cout << "original: " << img_inp[607498] << std::endl;
	std::cout << "cols: " << dst.cols << std::endl;
	std::cout << "rows: " << dst.rows << std::endl;
	printf("holaaaaaaaaaaaaaaaaaaaa6\n");
/* 	xf::Array2xfMat<INPUT_PTR_WIDTH,TYPE,HEIGHT,WIDTH,NPC_T>(img_inp,in_mat);
	xf::GaussianBlur<FILTER_SIZE,XF_BORDER_CONSTANT,TYPE,HEIGHT,WIDTH,NPC_T> (in_mat, out_mat, sigma);
	xf::xfMat2Array<OUTPUT_PTR_WIDTH,TYPE,HEIGHT,WIDTH,NPC_T>(out_mat,img_out); */


}
}
