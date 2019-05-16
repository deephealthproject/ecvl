#ifndef ECVL_IMGPROC_H_
#define ECVL_IMGPROC_H_

#include "core.h"
#include "support_opencv.h"

namespace ecvl {

enum class InterpolationType {
    nearest,
    linear,
    area,
    cubic,
    lanczos4
};

static int GetOpenCVInterpolation(InterpolationType interp);

void ResizeDim(const ecvl::Image& src, ecvl::Image& dst, const std::vector<int>& newdims, InterpolationType interp = InterpolationType::linear);
void ResizeScale(const ecvl::Image& src, ecvl::Image& dst, const std::vector<double>& scales, InterpolationType interp = InterpolationType::linear);

void Flip2D(const ecvl::Image& src, ecvl::Image& dst);

void Mirror2D(const ecvl::Image& src, ecvl::Image& dst);

void Rotate2D(const ecvl::Image& src, ecvl::Image& dst, double angle, const std::vector<double>& center = {}, double scale = 1.0, InterpolationType interp = InterpolationType::linear);

void RotateFullImage2D(const ecvl::Image& src, ecvl::Image& dst, double angle, double scale = 1.0, InterpolationType interp = InterpolationType::linear);

} // namespace ecvl

#endif // ECVL_IMGPROC_H_