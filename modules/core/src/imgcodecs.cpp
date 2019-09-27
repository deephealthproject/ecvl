#include "ecvl/core/imgcodecs.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ecvl/core/support_opencv.h"

#ifdef ECVL_WITH_DICOM
#include "ecvl/core/support_dcmtk.h"
#endif

using namespace std::filesystem;

namespace ecvl {

bool ImRead(const path& filename, Image& dst)
{
    dst = MatToImage(cv::imread(filename.string()));

#ifdef ECVL_WITH_DICOM
    if (dst.IsEmpty()) {
        DicomRead(filename.string(), dst);
    }
#endif

    return !dst.IsEmpty();
}

bool ImReadMulti(const std::string& filename, Image& dst) {

    std::vector<cv::Mat> v;
    cv::imreadmulti(filename, v);
    dst = MatVecToImage(v);

    return !dst.IsEmpty();
}

bool ImWrite(const path& filename, const Image& src)
{
    return cv::imwrite(filename.string(), ImageToMat(src));
}

} // namespace ecvl