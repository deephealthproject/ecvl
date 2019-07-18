#include "ecvl/core/support_nifti.h"

#include <string>
#include <fstream>

#include "ecvl/core/standard_errors.h"

//#include <nifti/nifti1_io.h>

using namespace std;

namespace ecvl {

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

    struct EndianReader {

        istream& is_;
        bool swap_endianness_;

        EndianReader(istream& is, bool swap_endianness = false) : is_(is), swap_endianness_(swap_endianness) {}

        void SwitchEndianness(char* data, size_t len) {
            for (size_t i = 0; i < len / 2; i++) {
                std::swap(data[i], data[len - i - 1]);
            }
        }

        void ReadSwapEndianness(char* dst, size_t len = 1, long long n = 1) {
            is_.read(dst, len * n);
            if (len > 1) {
                for (int i = 0; i < n; i++) {
                    SwitchEndianness(dst + i * len, len);
                }
            }
        }

        void operator()(char* dst, size_t len = 1, long long n = 1) {
            if (swap_endianness_) {
                ReadSwapEndianness(dst, len, n);
            }
            else {
                is_.read(dst, len * n);
            }
        }
    };

    // Home-made variation  
    bool NiftiRead(const std::string& filename, Image& dst) {

        ifstream is;
        is.exceptions(ifstream::failbit | ifstream::eofbit | ifstream::badbit);
        
        try {
        
            is.open(filename, ios::binary);

            // Nifti 1 header
            struct nifti {
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

                float srow_x[4];    /*!< 1st row affine transform.   */
                float srow_y[4];    /*!< 2nd row affine transform.   */
                float srow_z[4];    /*!< 3rd row affine transform.   */

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
                throw runtime_error("Wrong length of Nifti header file.");
            }
            EndianReader rd(is, swap_endianness);

            // Skip unused fields
            is.seekg(36u, ios::cur);

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

            // Skip some other data
            is.seekg(220u, ios::cur);

            is.read(reinterpret_cast<char*>(header.magic), sizeof(char) * 4);

            if (strcmp("ni1", header.magic) == 0) {
                string data_file = filename.substr(0, filename.find_last_of('.') + 1) + "img";
                is.close();
                is.open(data_file, ios::binary);
            }
            else if (strcmp("n+1", header.magic) == 0) {
                // Skip possible extension
                is.seekg(static_cast<size_t>(header.vox_offset));
            }
            else {
                throw std::ifstream::failure("Wrong magic string for Nifti.");
            }

            // Create ecvl::Image and read data

            // We only support images and volumes by now, not temporal series or other stuff
            if ((header.dim[0] < 2) || (header.dim[0] > 3 && !((header.dim[0] == 4) && header.dim[4] == 1))) {
                throw std::runtime_error("Unsupported number of dimensions.\n");
            }
            int ndims = header.dim[0];
            if (ndims == 4) {
                ndims = 3;
            }

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

            switch (color_type) {

            case ColorType::RGB:     dims.push_back(3);     channels += "xyzc";    break;
            case ColorType::GRAY:    dims.push_back(1);     channels += "xyzc";    break;
            default:                                                            break;

            }

            dst.Create(dims, data_type, channels, color_type, spacings);

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

    bool NiftiRead(const filesystem::path& filename, Image& dst) {
        return NiftiRead(filename.string(), dst);
    }

    struct ByteWriter {

        ostream& os_;

        ByteWriter(ostream& os) : os_(os) {}

        template <typename T>
        void operator()(const T& value, size_t len = sizeof(T)) {
            os_.write(reinterpret_cast<const char*>(&value), sizeof(T));
        }

        // Repeats the same character len times
        void Fill(size_t len, char c = 0) {
            for (size_t i = 0; i < len; i++) {
                os_.put(c);
            }
        }


    };

    bool NiftiWrite(const std::string& filename, const Image& src) {

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
                if (tmp1.channels_ != "cxy") {
                    RearrangeChannels(tmp1, tmp, "cxy");
                    use_tmp2 = true;
                }
            }
            else if (tmp1.channels_.size() == 4) {
                if (tmp1.channels_ != "cxyz") {
                    RearrangeChannels(tmp1, tmp, "cxyz");
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

            wr.Fill(36);

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

            wr(static_cast<float>(dims));
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

            wr.Fill(sizeof(float)* (7 - dims));

            wr(352.f);  // Only one .nii file with header and data. No header extension.

            wr.Fill(232);   // I don't use those fields for now

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

    bool NiftiWrite(const filesystem::path& filename, const Image& src) {
        return NiftiWrite(filename.string(), src);
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

    // Uses the library
    //bool OldNiftiRead(const std::string& filename, Image& dst) {

    //    nifti_image* nim = NULL;

    //    try {

    //        znzFile f = nifti_image_open(filename.c_str(), "r", &nim);
    //        if (f == NULL || nim == NULL) {
    //            throw std::runtime_error("Can't open nifti image.\n");
    //        }

    //        if (nifti_image_load(nim) != 0) {
    //            throw std::runtime_error("Can't load nifti pixel data.\n");
    //        }

    //        // We only support images and volumes by now, not temporal series or different stuff
    //        if ((nim->ndim < 2) || (nim->ndim > 3 && !(nim->ndim == 4) && nim->dim[3] == 1)) {
    //            throw std::runtime_error("Unsupported number of dimensions.\n");
    //        }
    //        int ndims = nim->ndim;
    //        if (nim->ndim == 4) {
    //            ndims = 3;
    //        }

    //        // Convert nifti_image into ecvl::Image
    //        std::vector<int> dims;
    //        std::vector<float> spacings;
    //        for (int i = 0; i < ndims; i++) {
    //            dims.push_back(nim->dim[i + 1]);
    //            spacings.push_back(nim->pixdim[i + 1]);
    //        }

    //        DataType data_type;
    //        switch (nim->datatype) {

    //        case  DT_BINARY:           data_type = DataType::uint8;     break;               /* binary (1 bit/voxel)         */                      // qua bisogna leggere bit a bit D:
    //        case  DT_UNSIGNED_CHAR:    data_type = DataType::uint8;     break;                      /* unsigned char (8 bits/voxel) */
    //        case  DT_SIGNED_SHORT:     data_type = DataType::int16;     break;                     /* signed short (16 bits/voxel) */
    //        case  DT_SIGNED_INT:       data_type = DataType::int32;     break;             /* signed int (32 bits/voxel)   */
    //        case  DT_FLOAT:            data_type = DataType::float32;   break;                /* float (32 bits/voxel)        */
    //        case  DT_DOUBLE:           data_type = DataType::float64;   break;                 /* double (64 bits/voxel)       */
    //        case  DT_RGB:              data_type = DataType::uint8;     break;            /* RGB triple (24 bits/voxel)   */                          // attenzione perché sono 3 canali
    //        case  DT_INT8:             data_type = DataType::int8;      break;            /* signed char (8 bits)         */
    //        case  DT_UINT16:           data_type = DataType::uint16;    break;                /* unsigned short (16 bits)     */
    //        //case  DT_UINT32:           data_type = DataType::uint32;    break;                /* unsigned int (32 bits)       */
    //        case  DT_INT64:            data_type = DataType::int64;     break;              /* long long (64 bits)          */
    //        //case  DT_UINT64:           data_type = DataType::uint64;    break;                /* unsigned long long (64 bits) */
    //        case  DT_RGBA32:           data_type = DataType::uint8;     break;               /* 4 byte RGBA (32 bits/voxel)  */                       // attenzione perché sono 4 canali
    //        //case  DT_COMPLEX256:       data_type = DataType::none;      break;                  /* long double pair (256 bits)  */                    // non supportato
    //        //case  DT_COMPLEX128:       data_type = DataType::none;      break;                  /* double pair (128 bits)       */                    // non supportato
    //        //case  DT_FLOAT128:         data_type = DataType::none;      break;                /* long double (128 bits)       */                      // non supportato
    //        //case  DT_UNKNOWN:          data_type = DataType::none;      break;               /* what it says, dude           */
    //        //case  DT_ALL:              data_type = DataType::none;      break;           /* not very useful (?)          */
    //        //case  DT_COMPLEX:          data_type = DataType::none;      break;               /* complex (64 bits/voxel)      */                       // non supportato
    //        default:                   throw runtime_error("Unsupported data type.\n");

    //        }

    //        ColorType color_type = ColorType::none;

    //        if (data_type != DataType::none) {
    //            switch (nim->datatype) {

    //            case DT_RGB:                    color_type = ColorType::RGB;    break;
    //            case DT_RGBA32:                 color_type = ColorType::RGB;    break;      // This should be RGBA but we don't have it!
    //            default:                        color_type = ColorType::GRAY;   break;

    //            }

    //        }

    //        string channels;

    //        switch (color_type) {

    //        case ColorType::RGB:     dims.push_back(3);     channels += "xyzc";    break;
    //        case ColorType::GRAY:    dims.push_back(1);     channels += "xyzc";    break;
    //        default:                                                            break;

    //        }

    //        dst.Create(dims, data_type, channels, color_type, spacings);

    //        // Copia i pixel
    //        if (nim->datatype == DT_BINARY) {
    //            // leggi bit a bit
    //            throw std::runtime_error("Not implemented.\n");
    //        }
    //        else if (nim->datatype == DT_RGB) {
    //            // leggi un piano alla volta
    //            for (int color = 0; color < 3; color++) {
    //                for (int i = 0; i < dst.datasize_ / 3; i++) {
    //                    dst.data_[color * (dst.datasize_ / 3) + i] = reinterpret_cast<uint8_t*>(nim->data)[i * 3 + color];
    //                }
    //            }
    //        }
    //        else if (nim->datatype == DT_RGBA32) {
    //            // leggi un piano alla volta, scartando alpha
    //            for (int color = 0; color < 3; color++) {
    //                for (int i = 0; i < dst.datasize_ / 3; i++) {
    //                    dst.data_[color * (dst.datasize_ / 3) + i] = reinterpret_cast<uint8_t*>(nim->data)[i * 4 + color];
    //                }
    //            }
    //        }
    //        else {

    //            memcpy(dst.data_, nim->data, dst.datasize_);

    //        }

    //        znzclose(f);
    //        nifti_image_free(nim);

    //        return true;
    //    }
    //    catch (const std::runtime_error&) {
    //        dst = Image();
    //        return false;
    //    }
    //}

} // namespace evl