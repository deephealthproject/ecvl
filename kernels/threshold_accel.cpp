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
// #pragma once
// #ifndef _XF_THRESHOLD_CONFIG_H_
// #define _XF_THRESHOLD_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_threshold.hpp"


/*  set the optimisation type*/


#define NO  1  // Normal Operation
#define RO  0 // Resource Optimized


/*  set the type of thresholding*/

#define THRESH_TYPE XF_THRESHOLD_TYPE_BINARY

#define INPUT_PTR_WIDTH  256
#define OUTPUT_PTR_WIDTH 256

typedef ap_uint<8>      ap_uint8_t;
typedef ap_uint<64>      ap_uint64_t;


/*  set the height and weight  */
#define HEIGHT 2160
#define WIDTH  3840

#if RO
#define NPIX				XF_NPPC8
#endif
#if NO
#define NPIX				XF_NPPC1
#endif



extern "C" {
static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(XF_8UC1, NPIX)) / 8) / (INPUT_PTR_WIDTH / 8);
void threshold_accel(ap_uint<INPUT_PTR_WIDTH> *img_inp, ap_uint<OUTPUT_PTR_WIDTH> *img_out,unsigned char thresh,unsigned char maxval,int rows, int cols)
{
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1 depth=__XF_DEPTH
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2 depth=__XF_DEPTH

#pragma HLS INTERFACE s_axilite port=thresh     
#pragma HLS INTERFACE s_axilite port=maxval     
#pragma HLS INTERFACE s_axilite port=rows     
#pragma HLS INTERFACE s_axilite port=cols    
#pragma HLS INTERFACE s_axilite port=return   
const int pROWS = HEIGHT;
const int pCOLS = WIDTH;
const int pNPC1 = NPIX;
xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPIX> in_mat(rows, cols);
xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPIX> out_mat(rows, cols);

#pragma HLS DATAFLOW


    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPIX>(img_inp, in_mat);

    xf::cv::Threshold<THRESH_TYPE, XF_8UC1, HEIGHT, WIDTH, NPIX>(in_mat, out_mat, thresh, maxval);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPIX>(out_mat, img_out);

}
}
