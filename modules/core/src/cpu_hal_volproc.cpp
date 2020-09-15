/*
* ECVL - European Computer Vision Library
* Version: 0.2.2
* copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/
#define _USE_MATH_DEFINES
#include "ecvl/core/cpu_hal.h"

#include <iostream>

#include "ecvl/core/imgproc.h"
using namespace std;

namespace ecvl
{
float SincNorm(float x)
{
    if (fabs(x) < 1e-7) {
        return 1.0f - fabs(x);
    }
    double y = M_PI * x;
    return static_cast<float>(sin(y) / y);
}

float Hanning(float x, int half_w)
{
    if (fabs(x) > half_w) {
        return 0;
    }
    else {
        return static_cast<float>(0.5 + 0.5 * cos(M_PI * x / half_w));
    }
}

Image SincKernel(const string& sincwindowtype, int w, int n)
{
    int nstore = n;
    if (nstore < 1) {
        nstore = 1;
    }
    Image ker({ nstore, 1, 1 }, DataType::float32, "xyc", ColorType::GRAY);
    auto it = ker.ContiguousBegin<float>(), end = ker.ContiguousEnd<float>();

    int half_w = (w - 1) / 2;
    float half_nstore = (nstore - 1) / 2.0f;

    for (int n = 0; n < nstore; ++n, ++it) {
        float x = (n - half_nstore) / half_nstore * half_w;
        if ((sincwindowtype == "hanning") || (sincwindowtype == "h")) {
            *it = SincNorm(x) * Hanning(x, half_w);
        }
        else {
            cerr << "ERROR: Unrecognised sinc window type - using hanning" << endl;
            ker = SincKernel("h", w, nstore);
            return ker;
        }
    }
    return ker;
}

float KernelVal(float x, int w, const Image& kernel)
{
  // linearly interpolates to get the kernel at the point (x)
  //   given the half-width w
    if (fabs(x) > w) return 0.0;
    float halfnk = (kernel.Width() - 1) / 2.0f;
    float dn = x / w * halfnk + halfnk + 1.0f;
    int n = (int)floor(dn);
    dn -= n;
    if (n > (kernel.Width() - 1)) return 0.0;
    if (n < 1) return 0.0;

    float* kernel_data = reinterpret_cast<float*>(kernel.data_);

    return  kernel_data[n - 1] * (1.0f - dn) + kernel_data[n] * dn;
}

inline bool InBounds(const int n_volumes, int index)
{
    return ((index >= 1) && (index <= n_volumes));
}

float Extrapolate(const Image& data, const int index)
{
    float extrapval;
    float* data_data = reinterpret_cast<float*>(data.data_);
    int n_volumes = data.dims_[3];

    if (InBounds(n_volumes, index))
        extrapval = data_data[index - 1];
    else if (InBounds(n_volumes, index - 1))
        extrapval = data_data[data.Width() - 1];
    else if (InBounds(n_volumes, index + 1))
        extrapval = data_data[0];
    else {
        auto it = data.Begin<float>();
        auto e = data.End<float>();
        double sum = 0;
        for (; it != e; ++it) {
            sum += *it;
        }
        extrapval = static_cast<float>(sum / std::accumulate(std::begin(data.dims_), std::end(data.dims_), 1, std::multiplies<int>()));
    }

    return extrapval;
}

float KernelInterpolation(const Image& data, float index, const Image& userkernel, int width)
{
    int widthx = (width - 1) / 2;
    // kernel half-width  (i.e. range is +/- w)
    int ix0;
    ix0 = (int)floor(index);
    int n_volumes = data.dims_[3];

    int wx(widthx);
    vector<float> storex(2 * wx + 1);
    for (int d = -wx; d <= wx; d++)
        storex[d + wx] = KernelVal((index - ix0 + d), wx, userkernel);

    float convsum = 0.0, interpval = 0.0, kersum = 0.0;

    int xj;
    float* data_data = reinterpret_cast<float*>(data.data_);

    for (int x1 = ix0 - wx; x1 <= ix0 + wx; x1++) {
        if (InBounds(n_volumes, x1)) {
            xj = ix0 - x1 + wx;
            float kerfac = storex[xj];
            convsum += data_data[x1 - 1] * kerfac;
            kersum += kerfac;
        }
    }

    if ((fabs(kersum) > 1e-9)) {
        interpval = convsum / kersum;
    }
    else {
        interpval = (float)Extrapolate(data, ix0);
    }
    return interpval;
}

void CpuHal::SliceTimingCorrection(const Image& src, Image& dst, bool odd, bool down)
{
    Image timeseries;
    ecvl::CopyImage(src, timeseries, DataType::float32);

    int no_volumes, no_slices;
    float repeat_time, offset = 0, slice_spacing;

    auto t = timeseries.channels_.find('t');
    no_volumes = timeseries.dims_[t];
    repeat_time = timeseries.spacings_[t];

    if (repeat_time == 0) {
        repeat_time = 3;
    }

    auto z = timeseries.channels_.find('z');
    no_slices = timeseries.dims_[z];
    slice_spacing = repeat_time / no_slices;

    Image userkernel = SincKernel("hanning", 7, 1201);

    float recenter = (((float)no_slices) / 2 - 0.5f) / no_slices;

    for (int slice = 1; slice <= no_slices; slice++) {
        if (odd) {
            if ((slice % 2) == 0)
                offset = recenter - (ceil((float)no_slices / 2) + ((slice - 1) / 2)) * (slice_spacing / repeat_time);
            else
                offset = recenter - ((slice - 1) / 2) * (slice_spacing / repeat_time);
        }
        else if (down) {
            offset = recenter - (no_slices - slice) * (slice_spacing / repeat_time);
        }
        else {
            offset = recenter - (slice - 1) * (slice_spacing / repeat_time);
        }

        for (int x_pos = 0; x_pos < timeseries.Width(); x_pos++)
            for (int y_pos = 0; y_pos < timeseries.Height(); y_pos++) {
                View<DataType::float32> tmp_v(timeseries, { 0,0,slice - 1,0 }, { -1,-1,-1,1 });
                View<DataType::float32> voxeltimeseries(timeseries, { x_pos, y_pos, slice - 1, 0 }, { 1, 1, 1, -1 });
                Image voxeltimeseries_image = voxeltimeseries;
                Image interpseries = voxeltimeseries;
                auto it_int = interpseries.ContiguousBegin<float>();
                for (int time_step = 1; time_step <= no_volumes; time_step++, ++it_int) {
                    *it_int = KernelInterpolation(voxeltimeseries_image, time_step + offset, userkernel, 7);
                }
                float* inter_data = reinterpret_cast<float*>(interpseries.data_);

                for (int t = 0; t < no_volumes; t++) {
                    *reinterpret_cast<float*>(timeseries.Ptr({ x_pos, y_pos, slice - 1, t })) = inter_data[t];
                }
            }
    }

    dst = std::move(timeseries);
}
} // namespace ecvl