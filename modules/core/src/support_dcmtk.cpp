#include "ecvl/core/support_dcmtk.h"

#include <dcmtk/dcmimage/diregist.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmjpeg/djdecode.h>

namespace ecvl {

bool DicomRead(const std::string& filename, Image& dst) {

    bool return_value = true;

    DJDecoderRegistration::registerCodecs();

    DicomImage* image = new DicomImage(filename.c_str());
    if (image == NULL) {
        return_value = false;
    }
    else {
        if (image->getStatus() == EIS_Normal) {
            const uint8_t* pixelData = reinterpret_cast<const uint8_t*> ((image->getOutputData(8 /* maybe it doesn't alyaws fit */, 0, 1)));
            if (pixelData != NULL) {
                /* do something useful with the pixel data */
                int x = static_cast<int>(image->getWidth());
                int y = static_cast<int>(image->getHeight());

                ColorType color_type = ColorType::RGB;
                int planes = 3;
                EP_Interpretation interpretation = image->getPhotometricInterpretation();
                if (interpretation == EP_Interpretation::EPI_Monochrome1 || interpretation == EP_Interpretation::EPI_Monochrome2) {
                    color_type = ColorType::GRAY;
                    planes = 1;
                }

                dst.Create({ x, y, planes }, ecvl::DataType::uint8, "xyc", color_type);
                memcpy(dst.data_, pixelData, x * y * planes);
            }
        }
        else {
            std::cerr << "Error: cannot load DICOM image (" << DicomImage::getString(image->getStatus()) << ")" << std::endl;
            return_value = false;
        }
    }
    delete image;

    if (!return_value) {
        dst = Image();
    }
    return return_value;
}

bool DicomRead(const filesystem::path& filename, Image& dst) {

    bool return_value = true;

    DJDecoderRegistration::registerCodecs();

    DicomImage* image = new DicomImage(filename.string().c_str());
    if (image == NULL) {
        return_value = false;
    }
    else {
        if (image->getStatus() == EIS_Normal) {
            const uint8_t* pixelData = reinterpret_cast<const uint8_t*> ((image->getOutputData(8 /* maybe it doesn't alyaws fit */, 0, 1)));
            if (pixelData != NULL) {
                /* do something useful with the pixel data */
                int x = static_cast<int>(image->getWidth());
                int y = static_cast<int>(image->getHeight());

                ColorType color_type = ColorType::RGB;
                int planes = 3;
                EP_Interpretation interpretation = image->getPhotometricInterpretation();
                if (interpretation == EP_Interpretation::EPI_Monochrome1 || interpretation == EP_Interpretation::EPI_Monochrome2) {
                    color_type = ColorType::GRAY;
                    planes = 1;
                }

                dst.Create({ x, y, planes }, ecvl::DataType::uint8, "xyc", color_type);
                memcpy(dst.data_, pixelData, x * y * planes);
            }
        }
        else {
            std::cerr << "Error: cannot load DICOM image (" << DicomImage::getString(image->getStatus()) << ")" << std::endl;
            return_value = false;
        }
    }
    delete image;

    if (!return_value) {
        dst = Image();
    }
    return return_value;
}

} // namespace ecvl