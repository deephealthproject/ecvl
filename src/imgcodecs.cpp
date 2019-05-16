#include "imgcodecs.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "support_opencv.h"

namespace ecvl {

bool ImRead(const std::string& filename, Image& dst)
{
    dst = MatToImage(cv::imread(filename));
    return !dst.IsEmpty();
}

bool ImRead(const filesystem::path& filename, Image& dst)
{
    return ImRead(filename.string(), dst);
}

bool ImWrite(const std::string& filename, const Image& src) 
{
    return cv::imwrite(filename, ImageToMat(src));
}

bool ImWrite(const filesystem::path& filename, const Image& src) 
{
    return ImWrite(filename.string(), src);
}

} // namespace ecvl