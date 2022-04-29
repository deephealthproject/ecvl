/*
* ECVL - European Computer Vision Library
* Version: 1.0.3
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#undef UNICODE
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

using namespace ecvl::filesystem;
using namespace std;

namespace ecvl
{
InitDCMTK::InitDCMTK()
{
    DJDecoderRegistration::registerCodecs();
}
InitDCMTK::~InitDCMTK()
{
    DJDecoderRegistration::cleanup();
}

void InsertTagValue(DcmElement* elem, Image& dst)
{
    auto tag = elem->getTag();
    string name = tag.getTagName();

    switch (tag.getVR().getEVR()) {
    case EVR_AE:
    case EVR_AS:
    case EVR_AT:
    case EVR_CS:
    case EVR_DA:
    case EVR_DS:
    case EVR_DT:
    case EVR_IS:
    case EVR_LO:
    case EVR_LT:
    case EVR_PN:
    case EVR_SH:
    case EVR_ST:
    case EVR_TM:
    case EVR_UI:
    case EVR_UT:
    {
        char* specific_val;
        elem->getString(specific_val);
        if (specific_val) {
            dst.meta_.insert({ name, MetaData(static_cast<string>(specific_val), 0) });
        }
        else {
            dst.meta_.insert({ name, MetaData("", 0) });
        }
        break;
    }
    case EVR_OB:
    case EVR_OW:
    case EVR_OD:
    case EVR_OF:
    {
        dst.meta_.insert({ name, MetaData("", 0) });
        break;
    }
    case EVR_FL:
    {
        float specific_val;
        elem->getFloat32(specific_val);
        dst.meta_.insert({ name, MetaData(specific_val, 0) });
        break;
    }
    case EVR_FD:
    {
        double specific_val;
        elem->getFloat64(specific_val);
        dst.meta_.insert({ name, MetaData(specific_val, 0) });
        break;
    }
    case EVR_SL:
    {
        int specific_val;
        elem->getSint32(specific_val);
        dst.meta_.insert({ name, MetaData(specific_val, 0) });
        break;
    }
    case EVR_SS:
    {
        short specific_val;
        elem->getSint16(specific_val);
        dst.meta_.insert({ name, MetaData(specific_val, 0) });
        break;
    }
    case EVR_UL:
    {
        unsigned int specific_val;
        elem->getUint32(specific_val);
        dst.meta_.insert({ name, MetaData(specific_val, 0) });
        break;
    }
    case EVR_US:
    {
        unsigned short specific_val;
        elem->getUint16(specific_val);
        dst.meta_.insert({ name, MetaData(specific_val, 0) });
        break;
    }
    case EVR_SQ:
    case EVR_UN:
    {
        ECVL_ERROR_NOT_IMPLEMENTED
            break;
    }
    default:
        ECVL_ERROR_NOT_REACHABLE_CODE
    }
}

bool DicomRead(const std::string& filename, Image& dst)
{
    static InitDCMTK init_dcmtk; // Created only first time DicomRead is called
    bool return_value = true;

    DicomImage* image = new DicomImage(filename.c_str());
    if (image == NULL) {
        return_value = false;
    }
    else {
        if (image->getStatus() == EIS_Normal) {
            // Read raw data
            const DiPixel* dipixel = image->getInterData();
            const void* dipixel_data = dipixel->getData();
            EP_Representation repr = dipixel->getRepresentation();

            int x = static_cast<int>(image->getWidth());
            int y = static_cast<int>(image->getHeight());

            ColorType color_type = ColorType::RGB;
            int planes = 3;
            EP_Interpretation interpretation = image->getPhotometricInterpretation();
            if (interpretation == EP_Interpretation::EPI_Monochrome1 || interpretation == EP_Interpretation::EPI_Monochrome2) {
                color_type = ColorType::GRAY;
                planes = 1;
            }

            DataType dst_datatype;
            switch (repr) {
            case EPR_Uint8: dst_datatype = DataType::uint8; break;
            case EPR_Sint8: dst_datatype = DataType::int8; break;
            case EPR_Uint16: dst_datatype = DataType::uint16; break;
            case EPR_Sint16: dst_datatype = DataType::int16; break;
            case EPR_Uint32: dst_datatype = DataType::int32; break;     // Risk of overflow. DataType::int64 could be used instead.
            case EPR_Sint32: dst_datatype = DataType::int32; break;
            default:    ECVL_ERROR_NOT_REACHABLE_CODE
            }

            dst.Create({ x, y, planes }, dst_datatype, "xyc", color_type);
            if (planes == 1) {
                memcpy(dst.data_, dipixel_data, x * y * DataTypeSize(dst_datatype));
            }
            else {
                for (int i = 0; i < planes; i++) {
                    memcpy(dst.data_ + x * y * DataTypeSize(dst_datatype) * i, reinterpret_cast<void* const*>(dipixel_data)[i], x * y * DataTypeSize(dst_datatype));
                }
            }

            // Read metadata, only of string type - not currently considered
            DcmFileFormat fileformat;
            OFCondition status = fileformat.loadFile(filename.c_str());
            if (status.good()) {
                DcmObject* object = nullptr;
                DcmElement* elem = nullptr;
                DcmDataset* dataset = fileformat.getDataset();
                DcmTag tag;
                // Check the amount of string copy. They can probably be reduced.
                string name, value;

                while (true) {
                    object = dataset->nextInContainer(object);
                    if (object == nullptr)
                        break;

                    elem = dynamic_cast<DcmElement*>(object);

                    InsertTagValue(elem, dst);
                }
            }

            // Read possible overlays - not currently considered
            //unsigned int overlay_count = image->getOverlayCount();
            //vector<Image> overlay_data;

            //for (unsigned int plane = 0; plane < overlay_count; ++plane) {
            //    unsigned int left_pos, right_pos, width, height;
            //    EM_Overlay mode;
            //    const void* overlay_pixels = image->getOverlayData(plane, left_pos, right_pos, width, height, mode, 0, 8, 1, 0, 0);
            //    //const void* full_overlay_data = image->getFullOverlayData(plane, width, height, mode);
            //    vector<int> dims({ static_cast<int>(width), static_cast<int>(height), 1 });
            //    overlay_data.emplace_back(dims, DataType::uint8, "xyc", ColorType::GRAY);
            //    memcpy(overlay_data.back().data_, overlay_pixels, width * height);
            //}
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

bool DicomRead(const path& filename, Image& dst)
{
    return DicomRead(filename.string(), dst);
}

template <class _RunIt>
static vector<uint8_t> CompressOverlay(_RunIt first, _RunIt last, size_t size)
{
    vector<uint8_t> out((size + 7) / 8, uint8_t(0));
    int i = 0;
    int pos = 0;
    for (auto it = first; it != last; ++it) {
        uint8_t x = *it;
        if (x != 0) {
            out[i] |= 1u << pos;
        }
        pos++;
        if (pos == 8) {
            pos = 0;
            i++;
        }
    }
    return out;
}

static vector<uint8_t> CompressOverlay(const Image& src)
{
    if (src.channels_ != "xyc" || src.colortype_ != ColorType::GRAY || src.elemsize_ != 1) {
        ECVL_ERROR_WRONG_PARAMS("src Image must have channels xyc, colortype GRAY and elemsize 1")
    }
    ConstView<DataType::uint8> v(src);
    return CompressOverlay(v.Begin(), v.End(), src.dims_[0] * src.dims_[1]);
}

bool DicomWrite(const std::string& filename, const Image& src)
{
    DcmFileFormat fileformat;
    DcmDataset* dataset = fileformat.getDataset();

    DcmTag tag;

    // Required tags
    char uid[100];
    dataset->putAndInsertString(DCM_SOPClassUID, UID_SecondaryCaptureImageStorage);
    dataset->putAndInsertString(DCM_SOPInstanceUID, dcmGenerateUniqueIdentifier(uid, SITE_INSTANCE_UID_ROOT));
    dataset->putAndInsertString(DCM_StudyInstanceUID, dcmGenerateUniqueIdentifier(uid, SITE_STUDY_UID_ROOT));
    dataset->putAndInsertString(DCM_SeriesInstanceUID, dcmGenerateUniqueIdentifier(uid, SITE_SERIES_UID_ROOT));
    dataset->putAndInsertString(DCM_Modality, "RTIMAGE");
    dataset->putAndInsertString(DCM_PatientID, "");
    dataset->putAndInsertString(DCM_PatientName, "");
    dataset->putAndInsertString(DCM_PatientBirthDate, "");
    dataset->putAndInsertString(DCM_PatientSex, "");

    // Insert metadata tags (strings only) - not currently considered
    //if (src.meta_) {
    //    DicomMetaData* metadata = dynamic_cast<DicomMetaData*>(src.meta_);
    //    for (const auto& x : metadata->tags) {
    //        long l;
    //        const char* str = x.first.c_str();
    //        char* str_end;
    //        errno = 0;
    //        l = strtol(str + 1, &str_end, 16);
    //        if (errno == ERANGE || str_end != str + 5) {
    //            continue;
    //        }
    //        uint16_t g = static_cast<uint16_t>(l);
    //        errno = 0;
    //        l = strtol(str + 6, &str_end, 16);
    //        if (errno == ERANGE || str_end != str + 10) {
    //            continue;
    //        }
    //        uint16_t e = static_cast<uint16_t>(l);
    //        DcmTagKey tag_key(g, e);
    //        dataset->putAndInsertString(tag_key, x.second.c_str());
    //    }
    //}

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

    // RGB Images
    dataset->putAndInsertUint16(DCM_SamplesPerPixel, samples_per_pixel);
    dataset->putAndInsertString(DCM_PhotometricInterpretation, photometric_interpretation.c_str(), static_cast<uint32_t>(photometric_interpretation.length()));
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

    // Examples of rescale options
    //dataset->putAndInsertString(DCM_RescaleIntercept, "-1000");
    //dataset->putAndInsertString(DCM_RescaleSlope, "1");
    //dataset->putAndInsertString(DCM_RescaleType, "HU");

    if (!src.contiguous_) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    if (!floating_point) {
        if (src.elemsize_ == 1) {
            dataset->putAndInsertUint8Array(DCM_PixelData, src.data_, static_cast<unsigned long>(src.datasize_));
        }
        else {
            dataset->putAndInsertUint16Array(DCM_PixelData, reinterpret_cast<const uint16_t*>(src.data_), static_cast<unsigned long>((src.datasize_ + 1) / 2));
        }
    }
    else if (src.elemtype_ == DataType::float32) {
        dataset->putAndInsertFloat32Array(DCM_FloatPixelData, reinterpret_cast<const float*>(src.data_), static_cast<unsigned long>(src.datasize_ / sizeof(float)));
    }
    else if (src.elemtype_ == DataType::float64) {
        dataset->putAndInsertFloat64Array(DCM_DoubleFloatPixelData, reinterpret_cast<const double*>(src.data_), static_cast<unsigned long>(src.datasize_ / sizeof(double)));
    }

    // Insert overlay if present in Image MetaData, under the name "overlay"
    if (!src.meta_.empty()) {
        try {
            Image overlay = any_cast<Image>(src.GetMeta("overlay").Get());
            string overlay_str;
            overlay_str.resize(overlay.datasize_);
            memcpy(const_cast<char*>(overlay_str.data()), overlay.data_, overlay.datasize_);

            DcmTagKey tag_key(0x6000, 0x0010);
            dataset->putAndInsertUint16(tag_key, src.dims_[1]);         // overlay image must be "xyc"

            tag_key.setElement(0x0011);
            dataset->putAndInsertUint16(tag_key, src.dims_[0]);

            tag_key.setElement(0x0040);
            dataset->putAndInsertString(tag_key, "G");

            int16_t origin[] = { 1, 1 };
            tag_key.setElement(0x0050);
            dataset->putAndInsertSint16Array(tag_key, origin, 2);

            tag_key.setElement(0x0100);
            dataset->putAndInsertUint16(tag_key, 1);

            tag_key.setElement(0x0102);
            dataset->putAndInsertUint16(tag_key, 0);

            auto compressed_overlay = CompressOverlay(overlay_str.begin(), overlay_str.end(), overlay_str.length());
            tag_key.setElement(0x3000);
            dataset->putAndInsertUint8Array(tag_key, compressed_overlay.data(), static_cast<const unsigned long>(compressed_overlay.size()));
        }
        catch (const exception&) {
            // Overlay not present in source Image
        }
    }

    // Overlay data
    //vector<Image> overlay_data;
    //vector<int> dims({ static_cast<int>(columns), static_cast<int>(rows), 1 });
    //overlay_data.emplace_back(dims, DataType::uint8, "xyc", ColorType::GRAY);
    //memset(overlay_data.back().data_, 0, overlay_data.back().datasize_);
    //overlay_data.back().data_[462 * 200 + 0] = 1;
    //overlay_data.emplace_back(dims, DataType::uint8, "xyc", ColorType::GRAY);
    //memset(overlay_data.back().data_, 0, overlay_data.back().datasize_);
    //overlay_data.back().data_[462 * 201 + 0] = 1;
    //int count = 0;
    //for (const auto& overlay : overlay_data) {
    //    DcmTagKey tag_key(0x6000 + count, 0x0010);
    //    dataset->putAndInsertUint16(tag_key, overlay.dims_[1]);         // overlay image must be "xyc"

    //    tag_key.setElement(0x0011);
    //    dataset->putAndInsertUint16(tag_key, overlay.dims_[0]);

    //    tag_key.setElement(0x0040);
    //    dataset->putAndInsertString(tag_key, "G");

    //    int16_t origin[] = { 1, 1 };
    //    tag_key.setElement(0x0050);
    //    dataset->putAndInsertSint16Array(tag_key, origin, 2);

    //    tag_key.setElement(0x0100);
    //    dataset->putAndInsertUint16(tag_key, 1);

    //    tag_key.setElement(0x0102);
    //    dataset->putAndInsertUint16(tag_key, 0);

    //    auto compressed_overlay = CompressOverlay(overlay);
    //    tag_key.setElement(0x3000);
    //    dataset->putAndInsertUint8Array(tag_key, compressed_overlay.data(), static_cast<const unsigned long>(compressed_overlay.size()));
    //    count += 2;
    //}

    OFCondition status = fileformat.saveFile(filename.c_str(), EXS_LittleEndianExplicit);
    if (status.bad())
        return false;

    return true;
}

bool DicomWrite(const path& filename, const Image& src)
{
    return DicomWrite(filename.string(), src);
}
} // namespace ecvl