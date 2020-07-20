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

#include "ap_int.h"
#include "hls_stream.h"
#include "assert.h"
#include "common/xf_common.h"
#include "common/xf_utility.h"
#include "imgproc/xf_warp_transform.hpp"
#include <iostream>

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

#define RGB 1

#define NO  1  // Normal Operation
#define RO  0 // Resource Optimized

/* Input image Dimensions */

#define WIDTH 			675	// Maximum Input image width
#define HEIGHT 			900   	// Maximum Input image height

/* Output image Dimensions */

#define NEWWIDTH 		1920  // Maximum output image width
#define NEWHEIGHT 		1800  // Maximum output image height


#if NO
#define NPC				XF_NPPC1
#endif

#if RGB
#define TYPE XF_8UC3
#define CH_TYPE XF_RGB
#else
#define TYPE XF_8UC1
#define CH_TYPE XF_GRAY
#endif

extern "C" {
    //using namespace std;
void warpTransform_accel(ap_uint<INPUT_PTR_WIDTH> *img_inp, ap_uint<OUTPUT_PTR_WIDTH> *img_out, int rows_in, int cols_in, int rows_out, int cols_out, float *P_matrix)
{
  #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2
  #pragma HLS INTERFACE m_axi     port=P_matrix  offset=slave bundle=gmem3
  #pragma HLS INTERFACE s_axilite port=img_inp               bundle=control
  #pragma HLS INTERFACE s_axilite port=img_out               bundle=control
  #pragma HLS INTERFACE s_axilite port=rows_in              bundle=control
  #pragma HLS INTERFACE s_axilite port=cols_in              bundle=control
  #pragma HLS INTERFACE s_axilite port=rows_out              bundle=control
  #pragma HLS INTERFACE s_axilite port=cols_out              bundle=control
  #pragma HLS INTERFACE s_axilite port=P_matrix              bundle=control
  //#pragma HLS INTERFACE s_axilite port=interp              bundle=control
  #pragma HLS INTERFACE s_axilite port=return                bundle=control


const int pROWS_INP = HEIGHT;
const int pCOLS_INP = WIDTH;
const int pROWS_OUT = NEWHEIGHT;
const int pCOLS_OUT = NEWWIDTH;
const int pNPC1 = NPC;
  //const int INTERPOLATION = interp;

  for (int i = 0; i < 9; i++)
      printf("%f\n", P_matrix[i]);



//cout << "P_matrix = " << endl << " "  << &P_matrix << endl << endl;
  xf::Mat<TYPE, NEWHEIGHT, NEWWIDTH, NPC> in_mat;

#pragma HLS stream variable=in_mat.data depth=pCOLS_INP/pNPC1
    in_mat.rows = rows_in;
  	in_mat.cols = cols_in;

    xf::Mat<TYPE, NEWHEIGHT, NEWWIDTH, NPC> out_mat;
#pragma HLS stream variable=out_mat.data depth=pCOLS_OUT/pNPC1

  	out_mat.rows = rows_out;
    out_mat.cols = cols_out;


#pragma HLS DATAFLOW
	printf("fin1\n");
    xf::Array2xfMat<INPUT_PTR_WIDTH,TYPE,NEWHEIGHT,NEWWIDTH,NPC>(img_inp,in_mat);
    //xf::warpTransform<HEIGHT,0,0,0,TYPE,HEIGHT,WIDTH,NPC, false>(in_mat, out_mat, P_matrix);
	printf("fin2\n");
    xf::warpTransform<300,150,0,0,TYPE,NEWHEIGHT,NEWWIDTH,NPC, false>(in_mat, out_mat, P_matrix);
	
	printf("fin3\n");
    xf::xfMat2Array<OUTPUT_PTR_WIDTH,TYPE,NEWHEIGHT,NEWWIDTH,NPC>(out_mat,img_out);
	 

}
}
