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
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_otsuthreshold.hpp"



//#include <time.h>


// port widths
#define INPUT_PTR_WIDTH  256
#define OUTPUT_PTR_WIDTH 256


/*  set the optimisation type*/


#define NO  1  // Normal Operation
#define RO  0 // Resource Optimized

/* Input image Dimensions */

#define WIDTH 			3840	// Maximum Input image width
#define HEIGHT 			2160   	// Maximum Input image height


// Resolve optimization type:
#if NO
#define NPC1 XF_NPPC1
#define PTR_WIDTH 8
#endif

#if RO
#define NPC1 XF_NPPC8
#define PTR_WIDTH 64
#endif

// Set the pixel depth:
#define TYPE XF_8UC1


extern "C" {
static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE, NPC1)) / 8) / (PTR_WIDTH / 8);
void otsuThreshold_accel(ap_uint<INPUT_PTR_WIDTH> *img_inp, int rows_in, int cols_in, uint8_t *thresh)
{
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem0 depth=__XF_DEPTH
#pragma HLS INTERFACE m_axi     port=thresh  offset=slave bundle=gmem1          
#pragma HLS INTERFACE s_axilite port=rows_in              
#pragma HLS INTERFACE s_axilite port=cols_in                         
#pragma HLS INTERFACE s_axilite port=return                


 xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> in_mat;

#pragma HLS DATAFLOW


	//clock_t t;

	xf::cv::Array2xfMat<INPUT_PTR_WIDTH,TYPE,HEIGHT,WIDTH,NPC1>(img_inp,in_mat);
	//t = clock();
  	xf::cv::OtsuThreshold<TYPE, HEIGHT, WIDTH, NPC1>(in_mat, *thresh);
	// t = clock() - t;
	// double time_taken = ((double)t)/CLOCKS_PER_SEC; // calculate the elapsed time
	// printf("Tiempo de ejecucion otsuThreshold_accel en FPGA: %f\n", time_taken);

}
}
