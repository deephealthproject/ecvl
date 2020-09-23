/*
* ECVL - European Computer Vision Library
* Version: 0.2.1
* copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include "ecvl/core/support_nifti.h"

#include <fstream>
#include <iostream>
#include <string>

#include "ecvl/core/filesystem.h"
#include "ecvl/core/standard_errors.h"

using namespace ecvl::filesystem;
using namespace std;

namespace ecvl
{
#define DT_NONE                    0
#define DT_UNKNOWN                 0     /* what it says, dude           */
#define DT_BINARY                  1     /* binary (1 bit/voxel)         */
#define DT_UNSIGNED_CHAR           2     /* unsigned char (8 bits/voxel) */
#define DT_SIGNED_SHORT            4     /* signed short (16 bits/voxel) */
#define DT_SIGNED_INT              8     /* signed int (32 bits/voxel)   */
#define DT_FLOAT                  16     /* float (32 bits/voxel)        */
#define DT_COMPLEX                32     /* complex (64 bits/voxel)      */
#define DT_DOUBLE                 64     /* double (64 bits/voxel)       */
#define DT_RGB                   128     /* RGB triple (24 bits/voxel)   */
#define DT_ALL                   255     /* not very useful (?)          */

    /*----- another set of names for the same ---*/
#define DT_UINT8                   2
#define DT_INT16                   4
#define DT_INT32                   8
#define DT_FLOAT32                16
#define DT_COMPLEX64              32
#define DT_FLOAT64                64
#define DT_RGB24                 128

                            /*------------------- new codes for NIFTI ---*/
#define DT_INT8                  256     /* signed char (8 bits)         */
#define DT_UINT16                512     /* unsigned short (16 bits)     */
#define DT_UINT32                768     /* unsigned int (32 bits)       */
#define DT_INT64                1024     /* long long (64 bits)          */
#define DT_UINT64               1280     /* unsigned long long (64 bits) */
#define DT_FLOAT128             1536     /* long double (128 bits)       */
#define DT_COMPLEX128           1792     /* double pair (128 bits)       */
#define DT_COMPLEX256           2048     /* long double pair (256 bits)  */
#define DT_RGBA32               2304     /* 4 byte RGBA (32 bits/voxel)  */

struct EndianReader
{
    istream& is_;
    bool swap_endianness_;

    EndianReader(istream& is, bool swap_endianness = false) : is_(is), swap_endianness_(swap_endianness) {}

    void SwitchEndianness(char* data, size_t len)
    {
        for (size_t i = 0; i < len / 2; i++) {
            std::swap(data[i], data[len - i - 1]);
        }
    }

    void ReadSwapEndianness(char* dst, size_t len = 1, long long n = 1)
    {
        is_.read(dst, len * n);
        if (len > 1) {
            for (int i = 0; i < n; i++) {
                SwitchEndianness(dst + i * len, len);
            }
        }
    }

    void operator()(char* dst, size_t len = 1, long long n = 1)
    {
        if (swap_endianness_) {
            ReadSwapEndianness(dst, len, n);
        }
        else {
            is_.read(dst, len * n);
        }
    }
};

// Home-made variation
bool NiftiRead(const path& filename, Image& dst)
{
    ifstream is;
    is.exceptions(ifstream::failbit | ifstream::eofbit | ifstream::badbit);

    try {
        is.open(filename, ios::binary);

        // Nifti 1 header
        struct nifti
        {
            int   sizeof_hdr;    /*!< MUST be 348           */  /* int sizeof_hdr;      */
            char  data_type[10]; /*!< ++UNUSED++            */  /* char data_type[10];  */
            char  db_name[18];   /*!< ++UNUSED++            */  /* char db_name[18];    */
            int   extents;       /*!< ++UNUSED++            */  /* int extents;         */
            short session_error; /*!< ++UNUSED++            */  /* short session_error; */
            char  regular;       /*!< ++UNUSED++            */  /* char regular;        */
            char  dim_info;      /*!< MRI slice ordering.   */  /* char hkey_un0;       */

                                                 /*--- was image_dimension substruct ---*/
            short dim[8];        /*!< Data array dimensions.*/  /* short dim[8];        */
            float intent_p1;    /*!< 1st intent parameter. */  /* short unused8;       */
                                                                /* short unused9;       */
            float intent_p2;    /*!< 2nd intent parameter. */  /* short unused10;      */
                                                                /* short unused11;      */
            float intent_p3;    /*!< 3rd intent parameter. */  /* short unused12;      */
                                                                /* short unused13;      */
            short intent_code;  /*!< NIFTI_INTENT_* code.  */  /* short unused14;      */
            short datatype;      /*!< Defines data type!    */  /* short datatype;      */
            short bitpix;        /*!< Number bits/voxel.    */  /* short bitpix;        */
            short slice_start;   /*!< First slice index.    */  /* short dim_un0;       */
            float pixdim[8];     /*!< Grid spacings.        */  /* float pixdim[8];     */
            float vox_offset;    /*!< Offset into .nii file */  /* float vox_offset;    */
            float scl_slope;    /*!< Data scaling: slope.  */  /* float funused1;      */
            float scl_inter;    /*!< Data scaling: offset. */  /* float funused2;      */
            short slice_end;     /*!< Last slice index.     */  /* float funused3;      */
            char  slice_code;   /*!< Slice timing order.   */
            char  xyzt_units;   /*!< Units of pixdim[1..4] */
            float cal_max;       /*!< Max display intensity */  /* float cal_max;       */
            float cal_min;       /*!< Min display intensity */  /* float cal_min;       */
            float slice_duration;/*!< Time for 1 slice.     */  /* float compressed;    */
            float toffset;       /*!< Time axis shift.      */  /* float verified;      */
            int   glmax;         /*!< ++UNUSED++            */  /* int glmax;           */
            int   glmin;         /*!< ++UNUSED++            */  /* int glmin;           */

                                                    /*--- was data_history substruct ---*/
            char  descrip[80];   /*!< any text you like.    */  /* char descrip[80];    */
            char  aux_file[24];  /*!< auxiliary filename.   */  /* char aux_file[24];   */

            short qform_code;   /*!< NIFTI_XFORM_* code.   */  /*-- all ANALYZE 7.5 ---*/
            short sform_code;   /*!< NIFTI_XFORM_* code.   */  /*   fields below here  */
                                                                /*   are replaced       */
            float quatern_b;    /*!< Quaternion b param.   */
            float quatern_c;    /*!< Quaternion c param.   */
            float quatern_d;    /*!< Quaternion d param.   */
            float qoffset_x;    /*!< Quaternion x shift.   */
            float qoffset_y;    /*!< Quaternion y shift.   */
            float qoffset_z;    /*!< Quaternion z shift.   */

            array<float, 4> srow_x;    /*!< 1st row affine transform.   */
            array<float, 4> srow_y;    /*!< 2nd row affine transform.   */
            array<float, 4> srow_z;    /*!< 3rd row affine transform.   */

            char intent_name[16];/*!< 'name' or meaning of data.  */

            char magic[4];      /*!< MUST be "ni1\0" or "n+1\0". */
        } header;

        /* Time to read fields */

        // This first fields allows us to understand wether we should switch the endianness or not
        bool swap_endianness = false;
        is.read(reinterpret_cast<char*>(&header.sizeof_hdr), sizeof(int));
        if (header.sizeof_hdr == 0x5C010000) {
            // Endianness is the contrary
            swap_endianness = true;
        }
        else if (header.sizeof_hdr == 0x0000015C);
        else {
            if (filename.extension() == ".nii") {
                std::cerr << ECVL_WARNING_MSG << "Wrong length of Nifti header file." << endl;
            }
            dst = Image();
            return false;
        }
        EndianReader rd(is, swap_endianness);

        // Skip unused fields
        is.seekg(35u, ios::cur);

        rd(reinterpret_cast<char*>(&header.dim_info), sizeof(char));
        rd(reinterpret_cast<char*>(header.dim), sizeof(short), 8);

        is.seekg(14u, ios::cur);

        rd(reinterpret_cast<char*>(&header.datatype), sizeof(short));

        // Non dovrebbe servire? Controlliamo però che sia coerente con datatype.
        rd(reinterpret_cast<char*>(&header.bitpix), sizeof(short));

        rd(reinterpret_cast<char*>(&header.slice_start), sizeof(short));

        rd(reinterpret_cast<char*>(header.pixdim), sizeof(float), 8);

        rd(reinterpret_cast<char*>(&header.vox_offset), sizeof(float));
        rd(reinterpret_cast<char*>(&header.scl_slope), sizeof(float));
        rd(reinterpret_cast<char*>(&header.scl_inter), sizeof(float));

        rd(reinterpret_cast<char*>(&header.slice_end), sizeof(short));
        rd(reinterpret_cast<char*>(&header.slice_code), sizeof(char));
        rd(reinterpret_cast<char*>(&header.xyzt_units), sizeof(char));

        is.seekg(8u, ios::cur);
        rd(reinterpret_cast<char*>(&header.slice_duration), sizeof(float));
        rd(reinterpret_cast<char*>(&header.toffset), sizeof(float));
        is.seekg(112u, ios::cur);

        rd(reinterpret_cast<char*>(&header.qform_code), sizeof(short));
        rd(reinterpret_cast<char*>(&header.sform_code), sizeof(short));
        rd(reinterpret_cast<char*>(&header.quatern_b), sizeof(float));
        rd(reinterpret_cast<char*>(&header.quatern_c), sizeof(float));
        rd(reinterpret_cast<char*>(&header.quatern_d), sizeof(float));
        rd(reinterpret_cast<char*>(&header.qoffset_x), sizeof(float));
        rd(reinterpret_cast<char*>(&header.qoffset_y), sizeof(float));
        rd(reinterpret_cast<char*>(&header.qoffset_z), sizeof(float));
        rd(reinterpret_cast<char*>(&header.srow_x), sizeof(float), 4);
        rd(reinterpret_cast<char*>(&header.srow_y), sizeof(float), 4);
        rd(reinterpret_cast<char*>(&header.srow_z), sizeof(float), 4);

        // Skip some other data
        is.seekg(16u, ios::cur);

        is.read(reinterpret_cast<char*>(header.magic), sizeof(char) * 4);

        if (strcmp("ni1", header.magic) == 0) {
            string data_file = filename.string().substr(0, filename.string().find_last_of('.') + 1) + "img";
            is.close();
            is.open(data_file, ios::binary);
        }
        else if (strcmp("n+1", header.magic) == 0) {
            // Skip possible extension
            is.seekg(static_cast<size_t>(header.vox_offset));
        }
        else {
            if (filename.extension() == ".nii") {
                std::cerr << ECVL_WARNING_MSG << "Wrong magic string for Nifti." << endl;
            }
            dst = Image();
            return false;
        }

        // Create ecvl::Image and read data
        int ndims = header.dim[0];

        // Convert nifti_image into ecvl::Image
        std::vector<int> dims;
        std::vector<float> spacings;
        for (int i = 0; i < ndims; i++) {
            dims.push_back(header.dim[i + 1]);
            spacings.push_back(header.pixdim[i + 1]);
        }

        DataType data_type;
        switch (header.datatype) {
        case  DT_BINARY:           data_type = DataType::uint8;     break;               /* binary (1 bit/voxel)         */                      // qua bisogna leggere bit a bit D:
        case  DT_UNSIGNED_CHAR:    data_type = DataType::uint8;     break;                      /* unsigned char (8 bits/voxel) */
        case  DT_SIGNED_SHORT:     data_type = DataType::int16;     break;                     /* signed short (16 bits/voxel) */
        case  DT_SIGNED_INT:       data_type = DataType::int32;     break;             /* signed int (32 bits/voxel)   */
        case  DT_FLOAT:            data_type = DataType::float32;   break;                /* float (32 bits/voxel)        */
        case  DT_DOUBLE:           data_type = DataType::float64;   break;                 /* double (64 bits/voxel)       */
        case  DT_RGB:              data_type = DataType::uint8;     break;            /* RGB triple (24 bits/voxel)   */                          // attenzione perché sono 3 canali
        case  DT_INT8:             data_type = DataType::int8;      break;            /* signed char (8 bits)         */
        case  DT_UINT16:           data_type = DataType::uint16;    break;                /* unsigned short (16 bits)     */
        //case  DT_UINT32:           data_type = DataType::uint32;    break;                /* unsigned int (32 bits)       */
        case  DT_INT64:            data_type = DataType::int64;     break;              /* long long (64 bits)          */
        //case  DT_UINT64:           data_type = DataType::uint64;    break;                /* unsigned long long (64 bits) */
        case  DT_RGBA32:           data_type = DataType::uint8;     break;               /* 4 byte RGBA (32 bits/voxel)  */                       // attenzione perché sono 4 canali
        //case  DT_COMPLEX256:       data_type = DataType::none;      break;                  /* long double pair (256 bits)  */                    // non supportato
        //case  DT_COMPLEX128:       data_type = DataType::none;      break;                  /* double pair (128 bits)       */                    // non supportato
        //case  DT_FLOAT128:         data_type = DataType::none;      break;                /* long double (128 bits)       */                      // non supportato
        //case  DT_UNKNOWN:          data_type = DataType::none;      break;               /* what it says, dude           */
        //case  DT_ALL:              data_type = DataType::none;      break;           /* not very useful (?)          */
        //case  DT_COMPLEX:          data_type = DataType::none;      break;               /* complex (64 bits/voxel)      */                       // non supportato
        default:                   throw runtime_error("Unsupported data type.\n");
        }

        ColorType color_type = ColorType::none;

        if (data_type != DataType::none) {
            switch (header.datatype) {
            case DT_RGB:                    color_type = ColorType::RGB;    break;
            case DT_RGBA32:                 color_type = ColorType::RGB;    break;      // This should be RGBA but we don't have it!
            default:                        color_type = ColorType::GRAY;   break;
            }
        }

        string channels;
        if (dims[ndims - 1] == 1 && ndims == 4) {
            channels = "xyz";
            dims.pop_back();
        }
        else if (ndims < 4) {
            channels = "xyz";
        }
        else {
            channels = "xyzt";
        }

        dst.Create(dims, data_type, channels, color_type, spacings);
        dst.meta_.map_.insert({ "xyzt_units", make_shared<MetaSingle<char>>(header.xyzt_units) });
        dst.meta_.map_.insert({ "dim_info", make_shared<MetaSingle<char>>(header.dim_info) });
        dst.meta_.map_.insert({ "slice_duration", make_shared<MetaSingle<float>>(header.slice_duration) });
        dst.meta_.map_.insert({ "toffset", make_shared<MetaSingle<float>>(header.toffset) });
        dst.meta_.map_.insert({ "qform_code", make_shared<MetaSingle<short>>(header.qform_code) });
        dst.meta_.map_.insert({ "sform_code", make_shared<MetaSingle<short>>(header.sform_code) });
        dst.meta_.map_.insert({ "quatern_b", make_shared<MetaSingle<float>>(header.quatern_b) });
        dst.meta_.map_.insert({ "quatern_c", make_shared<MetaSingle<float>>(header.quatern_c) });
        dst.meta_.map_.insert({ "quatern_d", make_shared<MetaSingle<float>>(header.quatern_d) });
        dst.meta_.map_.insert({ "qoffset_x", make_shared<MetaSingle<float>>(header.qoffset_x) });
        dst.meta_.map_.insert({ "qoffset_y", make_shared<MetaSingle<float>>(header.qoffset_y) });
        dst.meta_.map_.insert({ "qoffset_z", make_shared<MetaSingle<float>>(header.qoffset_z) });
        dst.meta_.map_.insert({ "srow_x", make_shared<MetaSingle<array<float, 4>>>(header.srow_x) });
        dst.meta_.map_.insert({ "srow_y", make_shared<MetaSingle<array<float, 4>>>(header.srow_y) });
        dst.meta_.map_.insert({ "srow_z", make_shared<MetaSingle<array<float, 4>>>(header.srow_z) });

        // Read data
        char* data;
        bool allocated_data = false;

        if (header.datatype == DT_BINARY || header.datatype == DT_RGB || header.datatype == DT_RGBA32) {
            size_t total_number_bytes = 1;
            for (int i = 1; i <= header.dim[0]; i++) {
                total_number_bytes *= header.dim[i];
            }
            total_number_bytes *= header.bitpix;
            total_number_bytes /= 8;
            data = new char[total_number_bytes];
            allocated_data = true;
            is.read(data, total_number_bytes);
        }

        // Copia i pixel
        if (header.datatype == DT_BINARY) {
            // leggi bit a bit
            throw std::runtime_error("Not implemented.\n");
        }
        else if (header.datatype == DT_RGB) {
            // leggi un piano alla volta

            for (int color = 0; color < 3; color++) {
                for (size_t i = 0; i < dst.datasize_ / 3; i++) {
                    dst.data_[color * (dst.datasize_ / 3) + i] = reinterpret_cast<uint8_t*>(data)[i * 3 + color];
                }
            }
        }
        else if (header.datatype == DT_RGBA32) {
            // leggi un piano alla volta, scartando alpha
            for (int color = 0; color < 3; color++) {
                for (size_t i = 0; i < dst.datasize_ / 3; i++) {
                    dst.data_[color * (dst.datasize_ / 3) + i] = reinterpret_cast<uint8_t*>(data)[i * 4 + color];
                }
            }
        }
        else {
            is.read(reinterpret_cast<char*>(dst.data_), dst.datasize_);
        }

        is.close();

        if (allocated_data) {
            delete[] data;
        }
    }
    catch (const ifstream::failure&) {
        dst = Image();
        return false;
    }
    catch (const runtime_error&) {
        dst = Image();
        return false;
    }

    return true;
}

struct ByteWriter
{
    ostream& os_;

    ByteWriter(ostream& os) : os_(os) {}

    template <typename T>
    void operator()(const T& value, size_t len = sizeof(T))
    {
        os_.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    // Repeats the same character len times
    void Fill(size_t len, char c = 0)
    {
        for (size_t i = 0; i < len; i++) {
            os_.put(c);
        }
    }
};

bool NiftiWrite(const path& filename, const Image& src)
{
    Image tmp;
    bool use_tmp1 = false;

    if (src.colortype_ == ColorType::RGB || src.colortype_ == ColorType::RGBA) {
        if (src.elemsize_ != 1) {
            CopyImage(src, tmp, DataType::uint8);
            use_tmp1 = true;
        }
    }

    const Image& tmp1 = use_tmp1 ? tmp : src;
    bool use_tmp2 = false;

    if (tmp1.colortype_ == ColorType::GRAY); // nothing to do
    else if (tmp1.colortype_ == ColorType::RGB || tmp1.colortype_ == ColorType::RGBA) {
        if (tmp1.channels_.size() == 3) {
            if (tmp1.channels_ != "zxy") {
                RearrangeChannels(tmp1, tmp, "zxy");
                use_tmp2 = true;
            }
        }
        else if (tmp1.channels_.size() == 4) {
            if (tmp1.channels_ != "zxyt") {
                RearrangeChannels(tmp1, tmp, "zxyt");
                use_tmp2 = true;
            }
        }
        else {
            ECVL_ERROR_NOT_IMPLEMENTED
        }
    }
    else {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    const Image& img = use_tmp2 ? tmp : tmp1;

    ofstream os;
    os.exceptions(ifstream::failbit | ifstream::eofbit | ifstream::badbit);

    try {
        os.open(filename, ios::binary);

        // Header fields
        ByteWriter wr(os);

        wr(348);

        wr.Fill(35);

        wr(img.meta_.map_.at("dim_info")->Query<char>());

        short dims = static_cast<short>(img.dims_.size());
        if (img.channels_.find('c') != string::npos) {
            dims--; // The color dimension mustn't be counted
        }

        wr(dims);
        for (size_t i = 0; i < img.dims_.size(); i++) {
            if (img.channels_[i] != 'c') {
                wr(static_cast<short>(img.dims_[i]));
            }
        }
        wr.Fill(sizeof(short) * (7 - dims));

        // Intent is optional
        wr.Fill(sizeof(float) * 3 + sizeof(short));

        short datatype;
        if (img.colortype_ == ColorType::RGB) {
            datatype = DT_RGB24;
        }
        else if (img.colortype_ == ColorType::RGBA) {
            datatype = DT_RGBA32;
        }
        else if (img.colortype_ == ColorType::GRAY) {
            switch (img.elemtype_) {
            case DataType::uint8:   datatype = DT_UINT8; break;
            case DataType::uint16:   datatype = DT_UINT16; break;
            //case DataType::uint32:   datatype = DT_UINT32; break;
            //case DataType::uint64:   datatype = DT_UINT64; break;
            case DataType::int8:   datatype = DT_INT8; break;
            case DataType::int16:   datatype = DT_INT16; break;
            case DataType::int32:   datatype = DT_INT32; break;
            case DataType::int64:   datatype = DT_INT64; break;
            case DataType::float32:   datatype = DT_FLOAT32; break;
            case DataType::float64:   datatype = DT_FLOAT64; break;
            default: ECVL_ERROR_NOT_IMPLEMENTED
            }
        }
        else {
            ECVL_ERROR_NOT_IMPLEMENTED
        }
        wr(datatype);

        short bitpix;
        if (img.colortype_ == ColorType::RGB) {
            bitpix = 24;
        }
        else if (img.colortype_ == ColorType::RGBA) {
            bitpix = 32;
        }
        else {
            bitpix = img.elemsize_ * 8;
        }
        wr(bitpix);

        wr.Fill(sizeof(short)); // slice_start

        wr(static_cast<float>(dims)); //pixdim
        if (img.spacings_.size() != img.dims_.size()) {
            // Don't trust them
            for (int i = 0; i < dims; i++) {
                wr(static_cast<float>(1));
            }
        }
        else {
            for (const float& x : img.spacings_) {
                wr(x);
            }
        }

        wr.Fill(sizeof(float) * (7 - dims));

        wr(352.f);  // vox_offset: Only one .nii file with header and data. No header extension.

        wr.Fill(11);

        wr(img.meta_.map_.at("xyzt_units")->Query<char>());

        wr.Fill(8);
        wr(img.meta_.map_.at("slice_duration")->Query<float>());
        wr(img.meta_.map_.at("toffset")->Query<float>());

        wr.Fill(112);   // I don't use those fields for now
        wr(img.meta_.map_.at("qform_code")->Query<short>());
        wr(img.meta_.map_.at("sform_code")->Query<short>());
        wr(img.meta_.map_.at("quatern_b")->Query<float>());
        wr(img.meta_.map_.at("quatern_c")->Query<float>());
        wr(img.meta_.map_.at("quatern_d")->Query<float>());
        wr(img.meta_.map_.at("qoffset_x")->Query<float>());
        wr(img.meta_.map_.at("qoffset_y")->Query<float>());
        wr(img.meta_.map_.at("qoffset_z")->Query<float>());
        wr(img.meta_.map_.at("srow_x")->Query<array<float, 4>>());
        wr(img.meta_.map_.at("srow_y")->Query<array<float, 4>>());
        wr(img.meta_.map_.at("srow_z")->Query<array<float, 4>>());

        wr.Fill(16);

        const char magic[] = "n+1";
        os.write(magic, 4);

        wr.Fill(4);

        // Write pixel data
        os.write(reinterpret_cast<const char*>(img.data_), img.datasize_);

        os.close();
    }
    catch (const ofstream::failure&) {
        return false;
    }

    return true;
}

#undef DT_NONE
#undef DT_UNKNOWN
#undef DT_BINARY
#undef DT_UNSIGNED_CHAR
#undef DT_SIGNED_SHORT
#undef DT_SIGNED_INT
#undef DT_FLOAT
#undef DT_COMPLEX
#undef DT_DOUBLE
#undef DT_RGB
#undef DT_ALL
#undef DT_UINT8
#undef DT_INT16
#undef DT_INT32
#undef DT_FLOAT32
#undef DT_COMPLEX64
#undef DT_FLOAT64
#undef DT_RGB24
#undef DT_INT8
#undef DT_UINT16
#undef DT_UINT32
#undef DT_INT64
#undef DT_UINT64
#undef DT_FLOAT128
#undef DT_COMPLEX128
#undef DT_COMPLEX256
#undef DT_RGBA32
} // namespace evl