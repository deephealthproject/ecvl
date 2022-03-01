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

#include "common/xf_utility.hpp"
#include "common/xf_common.hpp"




#define WIDTH 			3840	// Maximum Input image width
#define HEIGHT 			2160   	// Maximum Input image height
#define MAXN             3840	
#define NO 1
#define RO 0

/*  set the input types  */

#define T_8U 1  // Input type of 8U
#define T_16U 0 // Input type of 16U
#define T_16S 0 // Input type of 16S
#define T_32S 0 // Input type of 32S

// Resolve pixel depth:
#if T_8U
#define TYPE XF_8UC3
#if NO
#define PTR_WIDTH 256
#else
#define PTR_WIDTH 256
#endif
#define INTYPE unsigned char
#endif
#if T_16U
#define TYPE XF_16UC1
#if NO
#define PTR_WIDTH 16
#else
#define PTR_WIDTH 128
#endif
#define INTYPE unsigned short
#endif
#if T_16S
#define TYPE XF_16SC1
#if NO
#define PTR_WIDTH 16
#else
#define PTR_WIDTH 128
#endif
#define INTYPE signed short
#endif
#if T_32S
#define TYPE XF_32SC1
#if NO
#define PTR_WIDTH 32
#else
#define PTR_WIDTH 256
#endif
#define INTYPE signed int
#endif

// Resolve optimization type:
#if NO
#define NPC1 XF_NPPC1
#endif

#if RO
#define NPC1 XF_NPPC8
#endif



//static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE, NPC1)) / 8) / (PTR_WIDTH / 8);

void min_max_loc_accel(ap_uint<PTR_WIDTH> *img_in, int rows, int cols, int *coordsX, int *coordsY, size_t n) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in          offset=slave  bundle=gmem0 
    #pragma HLS INTERFACE m_axi      port=coordsX       offset=slave  bundle=gmem1
    #pragma HLS INTERFACE m_axi      port=coordsY       offset=slave  bundle=gmem2
    #pragma HLS INTERFACE s_axilite  port=n  
    #pragma HLS INTERFACE s_axilite  port=cols 			
    #pragma HLS INTERFACE s_axilite  port=rows 			
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    // Local objects:
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgInput(rows, cols);
      // clang-format off
    #pragma HLS STREAM variable=imgInput.data depth=2


    int aux[MAXN];
    #pragma HLS DATAFLOW

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(img_in, imgInput);

    for(int i=0;i<n;i++){
        aux[i]=0; coordsX[i]=0; coordsY[i]=0;
     }                            

    for (int y = 0; y < rows; y++) { // rows
        for (int x = 0; x < cols; x++) { // cols
            int value = imgInput.read(y*cols + x);
            for(int j=0; j<n; j++){
                if( value > aux[j] ){
                    for(int k=n-1; k>j; k--){
                        aux[k] = aux[k-1];
                        coordsX[k] = coordsX[k-1];
                        coordsY[k] = coordsY[k-1];
                    }
                    aux[j] = value;
                    coordsX[j] = x;
                    coordsY[j] = y;
                    break;
                }
            }
        }
    }


    return;
} // End of kernel
