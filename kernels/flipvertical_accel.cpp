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
#include "assert.h"
#include "common/xf_common.h"
#include "common/xf_utility.h"
#include  "hls_video.h"



extern "C" {
void flipvertical_accel(const unsigned int *img_inp, unsigned int *img_out,int rows_in, int cols_in)
{
	
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=img_inp               bundle=control
#pragma HLS INTERFACE s_axilite port=img_out               bundle=control
#pragma HLS INTERFACE s_axilite port=rows_in              bundle=control
#pragma HLS INTERFACE s_axilite port=cols_in              bundle=control
#pragma HLS INTERFACE s_axilite port=return                bundle=control


#pragma HLS INLINE
	for(int i = 0; i<rows_in;i++){
		for(int j = 0; j<cols_in;j++){
			//out_mat.write(i*cols_in +j,in_mat.data[((rows_in - 1)-i)*cols_in +j]);
			img_out[i*cols_in +j] = img_inp[((rows_in - 1)-i)*cols_in +j];
		}
	}
}
}
