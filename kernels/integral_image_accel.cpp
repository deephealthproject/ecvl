/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_integral_image.hpp"


#define NO 1 // Normal Operation (1-pixel implementation)

#define WIDTH 			3840	// Maximum Input image width
#define HEIGHT 			2160   	// Maximum Input image height

typedef unsigned short uint16_t;

#define NPC1 XF_NPPC1

#define INPUT_PTR_WIDTH 256
#define OUTPUT_PTR_WIDTH 256
//static constexpr int __XF_DEPTH_IN = (HEIGHT * WIDTH * (XF_PIXELWIDTH(XF_8UC1, NPC1)) / 8) / (INPUT_PTR_WIDTH / 8);
//static constexpr int __XF_DEPTH_OUT = (HEIGHT * WIDTH * (XF_PIXELWIDTH(XF_32UC1, NPC1)) / 8) / (OUTPUT_PTR_WIDTH / 8);

void integral_image_accel(ap_uint<INPUT_PTR_WIDTH> *img_inp, ap_uint<OUTPUT_PTR_WIDTH> *img_out, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1 
    #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2 
    #pragma HLS INTERFACE s_axilite port=rows     
    #pragma HLS INTERFACE s_axilite port=cols     
    #pragma HLS INTERFACE s_axilite port=return   
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> in_mat(rows, cols);
// clang-format off
    #pragma HLS stream variable=in_mat.data depth=2
    // clang-format on

    xf::cv::Mat<XF_32UC1, HEIGHT, WIDTH, NPC1> out_mat(rows, cols);
// clang-format off
    #pragma HLS stream variable=out_mat.data depth=2
// clang-format on

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(img_inp, in_mat);

    xf::cv::integral<XF_8UC1, XF_32UC1, HEIGHT, WIDTH, XF_NPPC1>(in_mat, out_mat);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_32UC1, HEIGHT, WIDTH, NPC1>(out_mat, img_out);
}
