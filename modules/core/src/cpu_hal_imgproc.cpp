/*
* ECVL - European Computer Vision Library
* Version: 0.1
* copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include "ecvl/core/cpu_hal.h"

#include <random>

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include "ecvl/core/image.h"
#include "ecvl/core/imgproc.h"
#include "ecvl/core/saturate_cast.h"
#include "ecvl/core/support_opencv.h"

using namespace std;

namespace ecvl
{
void CpuHal::Flip2D(const ecvl::Image& src, ecvl::Image& dst)
{
    size_t c_pos = src.channels_.find('c');
    size_t x_pos = src.channels_.find('x');
    size_t y_pos = src.channels_.find('y');

    if (c_pos == string::npos || x_pos == string::npos || y_pos == string::npos) {
        ECVL_ERROR_WRONG_PARAMS("Malformed src image")
    }

    int src_width = src.Width();
    int src_height = src.Height();
    int src_channels = src.Channels();

    int src_stride_c = src.strides_[c_pos];
    int src_stride_x = src.strides_[x_pos];
    int src_stride_y = src.strides_[y_pos];
    vector<uint8_t*> src_vch(src_channels), dst_vch(src_channels);

    // Get the pointers to channels starting pixels
    for (int i = 0; i < src_channels; ++i) {
        src_vch[i] = src.data_ + i * src_stride_c;
        dst_vch[i] = dst.data_ + i * src_stride_c;
    }

    int pivot = (src_height + 1) / 2;
    for (int r = 0, r_end = src_height - 1; r < pivot; ++r, --r_end) {
        // Get the address of next row
        int r1 = r * src_stride_y;
        int r2 = r_end * src_stride_y;
        for (int c = 0; c < src_width; ++c) {
            // Get the address of pixels in current row
            int p1 = r1 + src_stride_x * c;
            int p2 = r2 + src_stride_x * c;

#define ECVL_TUPLE(type, ...) \
        case DataType::type: \
            for (int ch = 0; ch < src_channels; ++ch) { \
                *reinterpret_cast<TypeInfo_t<DataType::type>*>(dst_vch[ch] + p1) = *reinterpret_cast<TypeInfo_t<DataType::type>*>(src_vch[ch] + p2); \
                *reinterpret_cast<TypeInfo_t<DataType::type>*>(dst_vch[ch] + p2) = *reinterpret_cast<TypeInfo_t<DataType::type>*>(src_vch[ch] + p1); \
            } \
            break;

            switch (src.elemtype_) {
#include "ecvl/core/datatype_existing_tuples.inc.h"
            }

#undef ECVL_TUPLE
        }
    }
}

void CpuHal::Mirror2D(const ecvl::Image& src, ecvl::Image& dst)
{
    size_t c_pos = src.channels_.find('c');
    size_t x_pos = src.channels_.find('x');
    size_t y_pos = src.channels_.find('y');

    if (c_pos == string::npos || x_pos == string::npos || y_pos == string::npos) {
        ECVL_ERROR_WRONG_PARAMS("Malformed src image")
    }

    int src_width = src.Width();
    int src_height = src.Height();
    int src_channels = src.Channels();

    int src_stride_c = src.strides_[c_pos];
    int src_stride_x = src.strides_[x_pos];
    int src_stride_y = src.strides_[y_pos];
    vector<uint8_t*> src_vch(src_channels), dst_vch(src_channels);

    // Get the pointers to channels starting pixels
    for (int i = 0; i < src_channels; ++i) {
        src_vch[i] = src.data_ + i * src_stride_c;
        dst_vch[i] = dst.data_ + i * src_stride_c;
    }

    int pivot = (src_width + 1) / 2;
    for (int r = 0; r < src_height; ++r) {
        // Get the address of next row
        int pos = r * src_stride_y;
        for (int c = 0, c_end = src_width - 1; c < pivot; ++c, --c_end) {
            // Get the address of pixels in current row
            int p1 = pos + src_stride_x * c;
            int p2 = pos + src_stride_x * c_end;

#define ECVL_TUPLE(type, ...) \
        case DataType::type: \
            for (int ch = 0; ch < src_channels; ++ch) { \
                *reinterpret_cast<TypeInfo_t<DataType::type>*>(dst_vch[ch] + p1) = *reinterpret_cast<TypeInfo_t<DataType::type>*>(src_vch[ch] + p2); \
                *reinterpret_cast<TypeInfo_t<DataType::type>*>(dst_vch[ch] + p2) = *reinterpret_cast<TypeInfo_t<DataType::type>*>(src_vch[ch] + p1); \
            } \
            break;

            switch (src.elemtype_) {
#include "ecvl/core/datatype_existing_tuples.inc.h"
            }

#undef ECVL_TUPLE
        }
    }
}

std::vector<double> CpuHal::Histogram(const Image& src)
{
    std::vector<double> hist(256, 0);

    ConstView<DataType::uint8> view(src);
    for (auto it = view.Begin(); it != view.End(); ++it) {
        hist[*it]++;
    }

    int total_pixels = std::accumulate(src.dims_.begin(), src.dims_.end(), 1, std::multiplies<int>());
    for (auto it = hist.begin(); it < hist.end(); it++) {
        *it /= total_pixels;
    }

    return hist;
}

int CpuHal::OtsuThreshold(const Image& src)
{
    std::vector<double> hist = Histogram(src);

    double mu_t = 0;
    for (size_t i = 1; i < hist.size(); i++) {
        mu_t += hist[i] * i;
    }

    double w_k = 0;
    double mu_k = 0;
    double sigma_max = 0;
    int threshold = 0;
    int hsize = vsize(hist);
    for (int k = 0; k < hsize - 1; k++) {
        w_k += hist[k];
        mu_k += hist[k] * k;

        double sigma = ((mu_t * w_k - mu_k) * (mu_t * w_k - mu_k)) / (w_k * (1 - w_k));
        if (sigma > sigma_max) {
            sigma_max = sigma;
            threshold = k;
        }
    }

    return threshold;
}

void CpuHal::Filter2D(const Image& src, Image& dst, const Image& ker, DataType type)
{
    int hlf_width = ker.dims_[0] / 2;
    int hlf_height = ker.dims_[1] / 2;

    TypeInfo_t<DataType::float64>* ker_data = reinterpret_cast<TypeInfo_t<DataType::float64>*>(ker.data_);

    uint8_t* dst_ptr = dst.data_;

    //auto dst_it = dst.ContiguousBegin<TypeInfo_t<DataType::uint8>>();
    //auto dst_it_end = dst.ContiguousEnd<TypeInfo_t<DataType::uint8>>();

    TypeInfo_t<DataType::uint8>* src_data = reinterpret_cast<TypeInfo_t<DataType::uint8>*>(src.data_);
    for (int chan = 0; chan < dst.dims_[2]; chan++) {
        for (int r = 0; r < dst.dims_[1]; r++) {
            for (int c = 0; c < dst.dims_[0]; c++) {
                double acc = 0;
                int i = 0;
                for (int rk = 0; rk < ker.dims_[1]; rk++) {
                    for (int ck = 0; ck < ker.dims_[0]; ck++) {
                        int x = c + ck - hlf_width;
                        if (x < 0) x = 0; else if (x >= dst.dims_[0]) x = dst.dims_[0] - 1;

                        int y = r + rk - hlf_height;
                        if (y < 0) y = 0; else if (y >= dst.dims_[1]) y = dst.dims_[1] - 1;

                        acc += ker_data[i] * src_data[x + y * src.strides_[1]];

                        i++;
                    }
                }

#define ECVL_TUPLE(type, ...) \
case DataType::type: *reinterpret_cast<TypeInfo_t<DataType::type>*>(dst_ptr) = static_cast<TypeInfo_t<DataType::type>>(acc); break;

                switch (type) {
#include "ecvl/core/datatype_existing_tuples.inc.h"
                }

#undef ECVL_TUPLE

                dst_ptr += dst.elemsize_;
            }
        }

        src_data += src.strides_[2] / sizeof(*src_data);
    }
}

void CpuHal::AdditiveLaplaceNoise(const Image& src, Image& dst, double std_dev)
{
    random_device rd;
    mt19937 gen(rd());
    exponential_distribution<> dist(1 / std_dev);

    for (uint8_t* dst_ptr = dst.data_, *src_ptr = src.data_; dst_ptr < dst.data_ + dst.datasize_; dst_ptr += dst.elemsize_, src_ptr += src.elemsize_) {
        double exp1 = dist(gen);
        double exp2 = dist(gen);

        double noise = exp1 - exp2;

#define ECVL_TUPLE(type, ...) \
case DataType::type: *reinterpret_cast<TypeInfo_t<DataType::type>*>(dst_ptr) = saturate_cast<TypeInfo_t<DataType::type>>(noise + *reinterpret_cast<TypeInfo_t<DataType::type>*>(src_ptr)); break;

        switch (dst.elemtype_) {
#include "ecvl/core/datatype_existing_tuples.inc.h"
        }

#undef ECVL_TUPLE
    }
}
} // namespace ecvl