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
#include <ap_int.h>
#include "common/xf_common.hpp"
#include "common/xf_structs.hpp"
#include "common/xf_utility.hpp"

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
#if NO
#define NPC1 XF_NPPC1
#if GRAY
#define PTR_WIDTH 8
#else
#define PTR_WIDTH 32
#endif
#endif
#if RO
#define NPC1 XF_NPPC4
#if GRAY
#define PTR_WIDTH 32
#else
#define PTR_WIDTH 128
#endif
#endif

// Set the input and output pixel depth:
#if GRAY
#define TYPE XF_8UC1
#else
#define TYPE XF_8UC3
#endif


extern "C" {
	static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE, NPC1)) / 8) / (INPUT_PTR_WIDTH / 8);

	void flipvertical_accel(ap_uint<INPUT_PTR_WIDTH> *SrcPtr, ap_uint<INPUT_PTR_WIDTH> *DstPtr, int rows, int cols) {
		// clang-format off
		#pragma HLS INTERFACE m_axi port=SrcPtr offset=slave bundle=gmem0 depth=__XF_DEPTH
		#pragma HLS INTERFACE m_axi port=DstPtr offset=slave bundle=gmem1 depth=__XF_DEPTH
		#pragma HLS INTERFACE s_axilite port=SrcPtr               bundle=control
		#pragma HLS INTERFACE s_axilite port=DstPtr               bundle=control
		#pragma HLS INTERFACE s_axilite port=rows bundle=control
		#pragma HLS INTERFACE s_axilite port=cols	bundle=control
		#pragma HLS INTERFACE s_axilite port=return	bundle=control
		// clang-format on

		xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgInput(rows, cols);
    	xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgOutput(rows, cols);
	

		#pragma HLS DATAFLOW
		xf::cv::Array2xfMat<INPUT_PTR_WIDTH,TYPE,HEIGHT,WIDTH,NPC1>(SrcPtr,imgInput);
	

		for (int column = 0; column < cols; column++)
		{	
				
			//FlipColumn(matriz,res, column, rows_in, cols_in);
			int max = (rows * cols) - cols + column;
			for (int row = 0; row < rows; row++)
			{
				//out_mat[column + row * cols_in]= in_mat[max - (row * cols_in)];//comprobar con write de matrices de array2xfmat, intentar encontra run memcpy y preguntar jorge
				imgOutput.write(column + row * cols,imgInput.read(max - (row * cols)));
			}
			
		}
		
		xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH,TYPE,HEIGHT,WIDTH,NPC1>(imgOutput,DstPtr);
		

		return;
	} 
}
