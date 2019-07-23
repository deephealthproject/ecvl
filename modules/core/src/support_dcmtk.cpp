#include "ecvl/core/support_dcmtk.h"

#include <dcmtk/dcmimage/diregist.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmjpeg/djdecode.h>
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dctag.h>
#include <dcmtk/dcmdata/dctagkey.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcdefine.h>
#include <dcmtk/dcmdata/dcuid.h>

#include <ecvl/core/standard_errors.h>

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

    DJDecoderRegistration::cleanup();

    return return_value;
}

bool DicomRead(const filesystem::path& filename, Image& dst) {
    return DicomRead(filename.string(), dst);
}

bool DicomWrite(const std::string& filename, const Image& src) {

    char uid[100];
    DcmFileFormat fileformat;
    DcmDataset* dataset = fileformat.getDataset();

    DcmTag tag;

    dataset->putAndInsertString(DCM_SOPClassUID, UID_SecondaryCaptureImageStorage);
    dataset->putAndInsertString(DCM_SOPInstanceUID, dcmGenerateUniqueIdentifier(uid, SITE_INSTANCE_UID_ROOT));
    dataset->putAndInsertString(DCM_PatientName, "John^Doe");

    uint16_t samples_per_pixel, planar_configuration, rows, columns, pixel_representation;
    std::string photometric_interpretation;
    bool floating_point = false;

    if (src.elemtype_ == DataType::int8 || src.elemtype_ == DataType::int16 || src.elemtype_ == DataType::int32 || src.elemtype_ == DataType::int64) {
        pixel_representation = 1;
    }
    else if (src.elemtype_ == DataType::uint8 || src.elemtype_ == DataType::uint16 /*|| src.elemtype_ == DataType::uint32 || src.elemtype_ == DataType::uint64*/) {
        pixel_representation = 0;
    }
    else if (src.elemtype_ == DataType::float32 || src.elemtype_ == DataType::float64) {
        if (src.colortype_ != ColorType::GRAY) {
            ECVL_ERROR_NOT_IMPLEMENTED
        }
        floating_point = true;
    }

    if (src.colortype_ == ColorType::GRAY) {
        samples_per_pixel = 1;
        photometric_interpretation = "MONOCHROME2"; // MONOCHROME1: 0 = white    MONOCHROME2: 0 = black
    }
    else if (src.colortype_ == ColorType::RGB /*|| src.colortype_ == ColorType::BGR*/) {
        samples_per_pixel = 3;
        photometric_interpretation = "RGB";
    }
    else {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if (src.channels_ == "cxy") {
        rows = src.dims_[2];
        columns = src.dims_[1];
        planar_configuration = 0;
    }
    else if (src.channels_ == "xyc") {
        rows = src.dims_[1];
        columns = src.dims_[0];
        planar_configuration = 1;
    }
    else {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    // Prova per immagini RGB
    dataset->putAndInsertUint16(DCM_SamplesPerPixel, samples_per_pixel);
    dataset->putAndInsertString(DCM_PhotometricInterpretation, photometric_interpretation.c_str(), photometric_interpretation.length());
    if (samples_per_pixel > 1) {
        dataset->putAndInsertUint16(DCM_PlanarConfiguration, planar_configuration);
    }
    dataset->putAndInsertUint16(DCM_Rows, rows);
    dataset->putAndInsertUint16(DCM_Columns, columns);
    dataset->putAndInsertUint16(DCM_BitsAllocated, src.elemsize_ * 8);
    if (!floating_point) { 
        dataset->putAndInsertUint16(DCM_BitsStored, src.elemsize_ * 8); 
        dataset->putAndInsertUint16(DCM_HighBit, src.elemsize_ * 8 - 1);
        dataset->putAndInsertUint16(DCM_PixelRepresentation, pixel_representation);
    }
    dataset->putAndInsertString(DCM_LossyImageCompression, "00");

    if (!floating_point) {
        dataset->putAndInsertUint8Array(DCM_PixelData, src.data_, src.datasize_);
    }
    else if (src.elemtype_ == DataType::float32) {
        dataset->putAndInsertFloat32Array(DCM_FloatPixelData, reinterpret_cast<const float *>(src.data_), src.datasize_ / sizeof(float));
    }
    else if (src.elemtype_ == DataType::float64) {
        dataset->putAndInsertFloat64Array(DCM_DoubleFloatPixelData, reinterpret_cast<const double*>(src.data_), src.datasize_ / sizeof(double));
    }

    OFCondition status = fileformat.saveFile(filename.c_str(), EXS_LittleEndianExplicit);
    if (status.bad())
        return false;

    return true;

}

bool DicomWrite(const filesystem::path& filename, const Image& src) {
    return DicomWrite(filename.string(), src);
}

} // namespace ecvl