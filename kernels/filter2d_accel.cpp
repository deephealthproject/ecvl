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
#include "imgproc/xf_custom_convolution.hpp"


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

/*  specify the shift parameter */
#define SHIFT 48

/* Input image Dimensions */

#define WIDTH 			2160	// Maximum Input image width
#define HEIGHT 			3840   	// Maximum Input image height

/* Output image Dimensions */

#define NEWWIDTH 		1920  // Maximum output image width
#define NEWHEIGHT 		1800  // Maximum output image height

/*  specify the shift parameter */
#define SHIFT 48


#define FILTER_HEIGHT 3
#define FILTER_WIDTH 3


#define OUT_8U 1
#define OUT_16S 0

/* Interface types*/
// Resolve optimization type:
#if RO
#define NPC1 XF_NPPC8
#endif
#if NO
#define NPC1 XF_NPPC1
#endif

#if GRAY
// Resolve pixel depth:
#if OUT_8U
#define INTYPE XF_8UC1
#define OUTTYPE XF_8UC1
#if NO
#define INPUT_PTR_WIDTH 8
#define OUTPUT_PTR_WIDTH 8
#else
#define INPUT_PTR_WIDTH 64
#define OUTPUT_PTR_WIDTH 64
#endif
#endif

#if OUT_16S
#define INTYPE XF_8UC1
#define OUTTYPE XF_16SC1
#if NO
#define INPUT_PTR_WIDTH 8
#define OUTPUT_PTR_WIDTH 16
#else
#define INPUT_PTR_WIDTH 64
#define OUTPUT_PTR_WIDTH 128
#endif
#endif

#else
// Resolve pixel depth:
#if OUT_8U
#define INTYPE XF_8UC3
#define OUTTYPE XF_8UC3
#if NO
#define INPUT_PTR_WIDTH 32
#define OUTPUT_PTR_WIDTH 32
#else
#define INPUT_PTR_WIDTH 256
#define OUTPUT_PTR_WIDTH 256
#endif
#endif

#if OUT_16S
#define INTYPE XF_8UC3
#define OUTTYPE XF_16SC3
#if NO
#define INPUT_PTR_WIDTH 64
#define OUTPUT_PTR_WIDTH 64
#else
#define INPUT_PTR_WIDTH 512
#define OUTPUT_PTR_WIDTH 512
#endif
#endif

#endif


extern "C" {
	using namespace std;

//static constexpr int __XF_DEPTH_IN = (HEIGHT * WIDTH * (XF_PIXELWIDTH(INTYPE, NPC1)) / 8) / (INPUT_PTR_WIDTH / 8);
//static constexpr int __XF_DEPTH_OUT = (HEIGHT * WIDTH * (XF_PIXELWIDTH(OUTTYPE, NPC1)) / 8) / (OUTPUT_PTR_WIDTH / 8);
//static constexpr int __XF_DEPTH_FILTER = FILTER_HEIGHT * FILTER_WIDTH;
void filter2d_accel(ap_uint<INPUT_PTR_WIDTH> *img_in, ap_uint<OUTPUT_PTR_WIDTH> *img_out,int rows, int cols, short int *filter)
{
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0 
    #pragma HLS INTERFACE m_axi      port=filter        offset=slave  bundle=gmem1
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem2 
    #pragma HLS INTERFACE s_axilite  port=rows 			          
    #pragma HLS INTERFACE s_axilite  port=cols 			          
    #pragma HLS INTERFACE s_axilite  port=return 			          

 	xf::cv::Mat<INTYPE, HEIGHT, WIDTH, NPC1> imgInput(rows, cols);
    xf::cv::Mat<OUTTYPE, HEIGHT, WIDTH, NPC1> imgOutput(rows, cols);

    #pragma HLS STREAM variable=imgInput.data depth=2
    #pragma HLS STREAM variable=imgOutput.data depth=2


    printf("cols dentro wrapper: %d\n", cols);
	printf("rows dentro wrapper: %d\n", rows);
    #pragma HLS DATAFLOW
 	 // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, INTYPE, HEIGHT, WIDTH, NPC1>(img_in, imgInput);
	xf::cv::filter2D<XF_BORDER_CONSTANT, FILTER_WIDTH, FILTER_HEIGHT, INTYPE, OUTTYPE, HEIGHT, WIDTH, NPC1>(
        imgInput, imgOutput, filter, SHIFT);//if the filter is not float, shift is zero, in ascii value 48
	// Convert _dst xf::cv::Mat object to output array:
    
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUTTYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);
printf("acaba\n");

}
}
