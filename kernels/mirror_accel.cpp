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
#include "assert.h"
#include "common/xf_common.h"
#include "common/xf_utility.h"
#include  "hls_video.h"



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

#define HEIGHT 2160
#define WIDTH  3840

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
void mirror_accel(ap_uint<INPUT_PTR_WIDTH> *img_inp, ap_uint<OUTPUT_PTR_WIDTH> *img_out,int rows_in, int cols_in)
{
	
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=img_inp               bundle=control
#pragma HLS INTERFACE s_axilite port=img_out               bundle=control
#pragma HLS INTERFACE s_axilite port=rows_in              bundle=control
#pragma HLS INTERFACE s_axilite port=cols_in              bundle=control
#pragma HLS INTERFACE s_axilite port=return                bundle=control

	const int pROWS_INP = HEIGHT;
	const int pCOLS_INP = WIDTH;
	const int pROWS_OUT = NEWHEIGHT;
	const int pCOLS_OUT = NEWWIDTH;
	const int pNPC = NPC_T;
	

	xf::Mat<TYPE, HEIGHT, WIDTH, NPC_T> in_mat;
#pragma HLS stream variable=in_mat.data depth=pCOLS_INP/pNPC

	xf::Mat<TYPE,HEIGHT, WIDTH, NPC_T> out_mat;
#pragma HLS stream variable=out_mat.data depth=pCOLS_OUT/pNPC

	in_mat.rows = rows_in;  in_mat.cols = cols_in;
	out_mat.rows = rows_in;  out_mat.cols = cols_in;
	
	//xf::Array2xfMat<INPUT_PTR_WIDTH,TYPE,HEIGHT,WIDTH,NPC_T>(img_inp,in_mat);
	//unsigned int aux = in_mat.read(899*cols_in+674);
	printf("\n dentro del kernel after: %d\n", rows_in);

#pragma HLS DATAFLOW
	
	xf::Array2xfMat<INPUT_PTR_WIDTH,TYPE,HEIGHT,WIDTH,NPC_T>(img_inp,in_mat);
	//unsigned int aux = in_mat.data[899*cols_in+674];
	//printf("\n dentro del kernel after: %d\n", in_mat.data[0*cols_in+0]);
	//printf("\n dentro del kernel after: %d\n", aux);
	for(int i = 0; i<rows_in;i++){
		for(int j = 0; j<cols_in;j++){
			//unsigned int aux = in_mat.read(((rows_in - 1)-i)*cols_in +j);
			//out_mat.write(i*cols_in +j,in_mat.data[((rows_in - 1)-i)*cols_in +j]);
			out_mat.data[i*cols_in +j] = in_mat.data[i*cols_in + (cols_in - 1 -j)];
			//printf("i: %d, j: %d, pos: %d, aux: %d\n", i, j, ((rows_in - 1)-i)*cols_in +j, aux);
		}
	}
	xf::xfMat2Array<OUTPUT_PTR_WIDTH,TYPE,HEIGHT,WIDTH,NPC_T>(out_mat,img_out);
}
}
