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

#include "common/xf_types.h"
#include "imgproc/xf_otsuthreshold.hpp"
#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.h"
#include "common/xf_utility.h"
#include "imgproc/xf_threshold.hpp"


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


#if RO
#define NPC				XF_NPPC8
#endif
#if NO
#define NPC				XF_NPPC1
#endif


extern "C" {
void otsuThreshold_accel(ap_uint<INPUT_PTR_WIDTH> *img_inp, int rows_in, int cols_in, uint8_t *thresh)
{
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=thresh  offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=img_inp               bundle=control
#pragma HLS INTERFACE s_axilite port=rows_in              bundle=control
#pragma HLS INTERFACE s_axilite port=cols_in              bundle=control
#pragma HLS INTERFACE s_axilite port=thresh              bundle=control


	const int pROWS_INP = HEIGHT;
	const int pCOLS_INP = WIDTH;
  const int pNPC1 = NPC;

  xf::Mat<XF_8UC1, HEIGHT, WIDTH, NPC> in_mat;

#pragma HLS stream variable=in_mat.data depth=pCOLS_INP/pNPC1
  	in_mat.rows = rows_in;
  	in_mat.cols = cols_in;

#pragma HLS DATAFLOW


	//clock_t t;

	xf::Array2xfMat<INPUT_PTR_WIDTH,XF_8UC1,HEIGHT,WIDTH,NPC>(img_inp,in_mat);
	//t = clock();
  xf::OtsuThreshold<XF_8UC1, HEIGHT, WIDTH, NPC>(in_mat, *thresh);
	// t = clock() - t;
	// double time_taken = ((double)t)/CLOCKS_PER_SEC; // calculate the elapsed time
	// printf("Tiempo de ejecucion otsuThreshold_accel en FPGA: %f\n", time_taken);

}
}
