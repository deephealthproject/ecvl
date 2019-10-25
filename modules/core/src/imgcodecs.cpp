#include "ecvl/core/imgcodecs.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ecvl/core/support_nifti.h"
#include "ecvl/core/support_opencv.h"

#ifdef ECVL_WITH_DICOM
#include "ecvl/core/support_dcmtk.h"
#endif

using namespace std::filesystem;

namespace ecvl {

bool ImRead(const path& filename, Image& dst, ImageFormat f)
{
    switch (f)
    {
    case ImageFormat::DEFAULT:
        dst = MatToImage(cv::imread(filename.string()));
        break;
    case ImageFormat::NIFTI:
        NiftiRead(filename, dst);
        break;
#ifdef ECVL_WITH_DICOM
    case ImageFormat::DICOM:
        DicomRead(filename, dst);
        break;
#endif
    }

    return !dst.IsEmpty();
}

bool ImReadMulti(const std::string& filename, Image& dst) {

    std::vector<cv::Mat> v;
    cv::imreadmulti(filename, v);
    dst = MatVecToImage(v);

    return !dst.IsEmpty();
}

bool ImWrite(const path& filename, const Image& src, ImageFormat f)
{
    switch (f)
    {
    case ImageFormat::DEFAULT:
        return cv::imwrite(filename.string(), ImageToMat(src));
    case ImageFormat::NIFTI:
        return NiftiWrite(filename, src);
#ifdef ECVL_WITH_DICOM
    case ImageFormat::DICOM:
        return DicomWrite(filename, src);
#endif
    }
}

} // namespace ecvl