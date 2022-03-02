/*
* ECVL - European Computer Vision Library
* Version: 1.0.2
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
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

#define ECVL_ERROR_RESERVED_BITS throw std::runtime_error(ECVL_ERROR_MSG "Wrong reserved bits in gz decompression");

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

class bitreader
{
    std::istream& _is;
    unsigned char _buffer;
    int _bits;

    bitreader(const bitreader&);
    bitreader& operator= (const bitreader&);

public:
    bitreader(std::istream& is) : _is(is), _bits(8) {}

    operator std::istream& () { return _is; }

    unsigned read_bit()
    {
        if (_bits == 8) {
            _is.get(reinterpret_cast<char&>(_buffer));
            _bits = 0;
        }
        return (_buffer >> _bits++) & 1;
    }

    // Read from the bitstream the count bit specified and put them in the least significant bit of the result
    unsigned operator() (unsigned count)
    {
        unsigned u = 0;
        for (unsigned i = 0; i < count; ++i) {
            u |= read_bit() << i;
        }
        return u;
    }
};

template<uint16_t Size>
class huffman
{
public:
    array<pair<uint32_t, uint8_t>, Size> codes;

    huffman(const array<uint8_t, Size>& lengths)
    {
        auto max_length = *max_element(lengths.begin(), lengths.end());
        vector<uint32_t> bl_count(max_length + 1, 0);
        for (uint16_t i = 0; i < Size; ++i) {
            bl_count[lengths[i]]++;
        }

        uint32_t code = 0;
        bl_count[0] = 0;
        vector<uint32_t> next_code(max_length + 1, 0);

        for (uint8_t bits = 1; bits <= max_length; bits++) {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        for (uint16_t n = 0; n < Size; n++) {
            uint8_t len = lengths[n];
            if (len != 0) {
                codes[n] = make_pair(next_code[len], len);
                next_code[len]++;
            }
        }
    }

    uint16_t decode(bitreader& br)
    {
        uint32_t buffer = 0;
        uint8_t len = 0;
        while (1) {
            auto bit = br(1);
            buffer = (buffer << 1) | bit;
            ++len;
            auto res = find(codes.begin(), codes.end(), make_pair(buffer, len));
            if (res != codes.end()) {
                return static_cast<uint16_t>(res - codes.begin());
            }
        }
    }
};

// Home-made variation
bool NiftiRead(const path& filename, Image& dst)
{
    ifstream ifile;
    ifile.exceptions(ifstream::failbit | ifstream::eofbit | ifstream::badbit);

    try {
        ifile.open(filename, ios::binary);

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

        // gzip header
        struct gzip
        {
            char  id1;               /*!< MUST be 31       */
            unsigned char  id2;      /*!< MUST be 139      */
            char  cm;               /*!< Compression Method     */
            char  flg;              /*!< FLaGs */
            int   mtime;              /*!< Modification TIME */
            char  xfl;              /*!< eXtra FLags */
            char  os;              /*!< Operating System */
            short xlen;              /*!< eXtra LENgth */
            //short crc16;              /*!< CRC-16 */
            int   crc32;              /*!< CRC-32 */
            int   isize;              /*!< Input SIZE */
        } header_gz;

        /* Check if the file is compressed with gzip*/
        ifile.read(reinterpret_cast<char*>(&header_gz.id1), sizeof(char));
        ifile.read(reinterpret_cast<char*>(&header_gz.id2), sizeof(char));

        string output_data;
        bool is_compressed = false;
        if (header_gz.id1 == 0x1f && header_gz.id2 == 0x8b) {
            is_compressed = true;
            bool fhcrc = false, fextra = false, fname = false, fcomment = false, freserved = false; // ignoring ftext

            ifile.read(reinterpret_cast<char*>(&header_gz.cm), sizeof(char));
            ifile.read(reinterpret_cast<char*>(&header_gz.flg), sizeof(char));
            ifile.read(reinterpret_cast<char*>(&header_gz.mtime), sizeof(int));
            ifile.read(reinterpret_cast<char*>(&header_gz.xfl), sizeof(char));
            ifile.read(reinterpret_cast<char*>(&header_gz.os), sizeof(char));

            if (header_gz.cm != 8) {
                throw std::runtime_error(ECVL_ERROR_MSG "Wrong compression method in gz decompression");
            }

            //if (header_gz.cm >= 0 && header_gz.cm < 8) {
            //    throw std::runtime_error(ECVL_ERROR_MSG "Wrong reserved bits in gz decompression");
            //}

            fhcrc = header_gz.flg & 0b00000010;
            fextra = header_gz.flg & 0b00000100;
            fname = header_gz.flg & 0b00001000;
            fcomment = header_gz.flg & 0b00010000;
            freserved = header_gz.flg & 0b11100000;

            if (freserved) {
                ECVL_ERROR_RESERVED_BITS
            }

            if (fextra) {
                ifile.read(reinterpret_cast<char*>(&header_gz.xlen), sizeof(short));
                ifile.seekg(header_gz.xlen, ios::cur);
            }

            if (fname) {
                char c;
                while (1) {
                    ifile.get(c);
                    if (c == 0) {
                        break;
                    }
                }
            }

            if (fcomment) {
                char c;
                while (1) {
                    ifile.get(c);
                    if (c == 0) {
                        break;
                    }
                }
            }

            if (fhcrc) {
                ifile.seekg(16, ios::cur);
                //is.read(reinterpret_cast<char*>(&header_gz.crc16), sizeof(short));
            }

            int iter = 0;
            bitreader br(ifile);
            bool bfinal = false;

            // while it's not the last block
            do {
                bfinal = br.read_bit();
                uint8_t btype = br(2);

                if (btype == 3) {
                    // reserved - error
                    ECVL_ERROR_RESERVED_BITS
                }
                if (btype == 0) {
                    // no compression
                    short len;
                    ifile.read(reinterpret_cast<char*>(&len), sizeof(short));
                    ifile.seekg(2, ios::cur);
                    output_data.resize(output_data.size() + len);
                    ifile.read(reinterpret_cast<char*>(output_data.data() + (len * iter)), len);
                    ++iter;
                }
                else {  // 01 = compressed with fixed huffman code and 10 = compressed with dynamic huffman code
                    array<uint8_t, 288> lit_len_code_lengths{}; // automatically initialized to 0
                    array<uint8_t, 32> distance_code_lengths{};

                    if (btype == 2) {
                        //  dynamic - read representation of code trees
                        uint16_t hlit = br(5) + 257;
                        uint8_t hdist = br(5) + 1;
                        uint8_t hclen = br(4) + 4;

                        array<uint8_t, 19> cl_cl{ 0 }; // code lengths for the code length alphabet
                        array<uint8_t, 19> indices{ 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };

                        for (int i = 0; i < hclen; ++i) {
                            cl_cl[indices[i]] = br(3);
                        }

                        huffman<19> h(cl_cl);
                        vector<uint8_t> total_code_lengths(hlit + hdist, 0);
                        for (uint16_t i = 0; i < hlit + hdist;) {
                            auto value = h.decode(br);
                            switch (value) {
                            case 16:
                            {
                                auto repeat = br(2) + 3;
                                auto prev_length = total_code_lengths[i - 1];

                                for (unsigned j = 0; j < repeat; ++j) {
                                    total_code_lengths[i++] = prev_length;
                                }
                            }
                            break;
                            case 17:
                            {
                                auto repeat = br(3) + 3;
                                for (unsigned j = 0; j < repeat; ++j) {
                                    total_code_lengths[i++] = 0;
                                }
                            }
                            break;
                            case 18:
                            {
                                auto repeat = br(7) + 11;
                                for (unsigned j = 0; j < repeat; ++j) {
                                    total_code_lengths[i++] = 0;
                                }
                            }
                            break;
                            default:
                                total_code_lengths[i++] = static_cast<uint8_t>(value);
                            }
                        }

                        copy(total_code_lengths.begin(), total_code_lengths.begin() + hlit, lit_len_code_lengths.begin());
                        copy(total_code_lengths.begin() + hlit, total_code_lengths.end(), distance_code_lengths.begin());
                    }
                    else if (btype == 1) {
                        fill(lit_len_code_lengths.begin(), lit_len_code_lengths.begin() + 144, 8);
                        fill(lit_len_code_lengths.begin() + 144, lit_len_code_lengths.begin() + 256, 9);
                        fill(lit_len_code_lengths.begin() + 256, lit_len_code_lengths.begin() + 280, 7);
                        fill(lit_len_code_lengths.begin() + 280, lit_len_code_lengths.end(), 8);

                        distance_code_lengths.fill(5);
                    }
                    else {
                        ECVL_ERROR_NOT_REACHABLE_CODE
                    }

                    huffman<288> h_lit_len(lit_len_code_lengths);
                    huffman<32> h_dist(distance_code_lengths);

                    while (1) {
                        auto value = h_lit_len.decode(br);
                        if (value < 256) {
                            output_data.push_back(static_cast<uint8_t>(value));
                        }
                        else if (value == 256) {
                            break;
                        }
                        else {
                            uint16_t length;
                            if (value < 265) {
                                length = value - 254;
                            }
                            else if (value < 269) {
                                length = (value - 265) * 2 + 11 + br(1);
                            }
                            else if (value < 273) {
                                length = (value - 269) * 4 + 19 + br(2);
                            }
                            else if (value < 277) {
                                length = (value - 273) * 8 + 35 + br(3);
                            }
                            else if (value < 281) {
                                length = (value - 277) * 16 + 67 + br(4);
                            }
                            else if (value < 285) {
                                length = (value - 281) * 32 + 131 + br(5);
                            }
                            else if (value == 285) {
                                length = 258;
                            }
                            else {
                                ECVL_ERROR_NOT_REACHABLE_CODE
                            }

                            value = h_dist.decode(br);
                            uint16_t dist;
                            if (value < 4) {
                                dist = value + 1;
                            }
                            else if (value < 30) {
                                uint8_t extra_bits = (value - 2) / 2;
                                uint16_t power = 1 << extra_bits;
                                uint16_t start_value = power * 2 + 1;
                                dist = start_value + power * (value % 2) + br(extra_bits);
                            }
                            else {
                                ECVL_ERROR_NOT_REACHABLE_CODE
                            }

                            size_t start_pos = output_data.size() - dist;
                            for (uint16_t j = 0; j < length; ++j) {
                                output_data.push_back(output_data[start_pos + j]);
                            }
                        }
                    }
                }
            } while (!bfinal);

            ifile.read(reinterpret_cast<char*>(&header_gz.crc32), sizeof(int));
            ifile.read(reinterpret_cast<char*>(&header_gz.isize), sizeof(int));
            if (output_data.size() % (1ull << 32) != header_gz.isize) {
                throw std::runtime_error(ECVL_ERROR_MSG "Size of the original (uncompressed) input does not match isize");
            }
        }
        else {
            ifile.seekg(0, ios::beg);
        }
        stringstream ss(output_data);
        istream* is_ptr = is_compressed ? dynamic_cast<istream*>(&ss) : dynamic_cast<istream*>(&ifile);
        istream& is = *is_ptr;
        /* Time to read fields */

        // This first fields allows us to understand whether we should switch the endianness or not
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

        // This should not be useful
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
            if (is_compressed) {
                ECVL_ERROR_NOT_IMPLEMENTED
            }
            else {
                ifile.close();
                ifile.open(data_file, ios::binary);
            }
        }
        else if (strcmp("n+1", header.magic) == 0) {
            // Skip possible extension
            is.seekg(static_cast<size_t>(header.vox_offset));
        }
        else {
            if (filename.extension() == ".nii") {
                std::cerr << ECVL_WARNING_MSG << "Wrong magic string for NIfTI." << endl;
            }
            dst = Image();
            return false;
        }

        // Create ecvl::Image and read data
        int ndims = header.dim[0];

        // Convert nifti_image into ecvl::Image
        DataType data_type;
        switch (header.datatype) {
        case  DT_BINARY:           data_type = DataType::uint8;     break;               /* binary (1 bit/voxel)         */                      // bit-per-bit reading, not implemented
        case  DT_UNSIGNED_CHAR:    data_type = DataType::uint8;     break;                      /* unsigned char (8 bits/voxel) */
        case  DT_SIGNED_SHORT:     data_type = DataType::int16;     break;                     /* signed short (16 bits/voxel) */
        case  DT_SIGNED_INT:       data_type = DataType::int32;     break;             /* signed int (32 bits/voxel)   */
        case  DT_FLOAT:            data_type = DataType::float32;   break;                /* float (32 bits/voxel)        */
        case  DT_DOUBLE:           data_type = DataType::float64;   break;                 /* double (64 bits/voxel)       */
        case  DT_RGB:              data_type = DataType::uint8;     break;            /* RGB triple (24 bits/voxel)   */                          // warning: 3 channels
        case  DT_INT8:             data_type = DataType::int8;      break;            /* signed char (8 bits)         */
        case  DT_UINT16:           data_type = DataType::uint16;    break;                /* unsigned short (16 bits)     */
        //case  DT_UINT32:           data_type = DataType::uint32;    break;                /* unsigned int (32 bits)       */
        case  DT_INT64:            data_type = DataType::int64;     break;              /* long long (64 bits)          */
        //case  DT_UINT64:           data_type = DataType::uint64;    break;                /* unsigned long long (64 bits) */
        case  DT_RGBA32:           data_type = DataType::uint8;     break;               /* 4 byte RGBA (32 bits/voxel)  */                       // warning: 4 channels
        //case  DT_COMPLEX256:       data_type = DataType::none;      break;                  /* long double pair (256 bits)  */                    // unsupported
        //case  DT_COMPLEX128:       data_type = DataType::none;      break;                  /* double pair (128 bits)       */                    // unsupported
        //case  DT_FLOAT128:         data_type = DataType::none;      break;                /* long double (128 bits)       */                      // unsupported
        //case  DT_UNKNOWN:          data_type = DataType::none;      break;               /* what it says, dude           */
        //case  DT_ALL:              data_type = DataType::none;      break;           /* not very useful (?)          */
        //case  DT_COMPLEX:          data_type = DataType::none;      break;               /* complex (64 bits/voxel)      */                       // unsupported
        default:                   throw runtime_error("Unsupported data type.\n");
        }

        ColorType color_type = ColorType::none;

        if (data_type != DataType::none) {
            switch (header.datatype) {
            case DT_RGB:                    color_type = ColorType::RGB;    break;
            case DT_RGBA32:                 color_type = ColorType::RGBA;    break;      // This should be RGBA and we have it!
            default:                        color_type = ColorType::none;   break;
            }
        }

        std::vector<int> dims;
        std::vector<float> spacings;

        for (int i = 0; i < ndims; i++) {
            dims.push_back(header.dim[i + 1]);
            spacings.push_back(header.pixdim[i + 1]);
        }

        string possible_channels = "xyzt";
        string channels = "";
        for (int i = 0; i < ndims && i < 4; i++) {
            channels += possible_channels[i];
        }
        for (int i = 4; i < ndims; i++) {
            channels += "o";
        }
        if (dims[ndims - 1] == 1) {
            dims.pop_back();
            channels.pop_back();
            spacings.pop_back();
        }

        if (color_type == ColorType::RGB) {
            dims.push_back(3);
            spacings.push_back(1);
        }
        if (color_type == ColorType::RGBA) {
            dims.push_back(4);
            spacings.push_back(1);
        }
        if (color_type == ColorType::RGB || color_type == ColorType::RGBA) {
            channels += "c";
        }

        dst.Create(dims, data_type, channels, color_type, spacings);
        dst.meta_.insert({ "xyzt_units", MetaData(header.xyzt_units, 0) });
        dst.meta_.insert({ "dim_info", MetaData(header.dim_info, 0) });
        dst.meta_.insert({ "slice_duration", MetaData(header.slice_duration, 0) });
        dst.meta_.insert({ "toffset", MetaData(header.toffset, 0) });
        dst.meta_.insert({ "qform_code", MetaData(header.qform_code, 0) });
        dst.meta_.insert({ "sform_code", MetaData(header.sform_code, 0) });
        dst.meta_.insert({ "quatern_b", MetaData(header.quatern_b, 0) });
        dst.meta_.insert({ "quatern_c", MetaData(header.quatern_c, 0) });
        dst.meta_.insert({ "quatern_d", MetaData(header.quatern_d, 0) });
        dst.meta_.insert({ "qoffset_x", MetaData(header.qoffset_x, 0) });
        dst.meta_.insert({ "qoffset_y", MetaData(header.qoffset_y, 0) });
        dst.meta_.insert({ "qoffset_z", MetaData(header.qoffset_z, 0) });
        dst.meta_.insert({ "srow_x", MetaData(header.srow_x, 0) });
        dst.meta_.insert({ "srow_y", MetaData(header.srow_y, 0) });
        dst.meta_.insert({ "srow_z", MetaData(header.srow_z, 0) });

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

        // Count pixels
        if (header.datatype == DT_BINARY) {
            // Read bit per bit
            throw std::runtime_error("Support for binary images is not implemented.\n");
        }
        else if (header.datatype == DT_RGB) {
            // Read a plane at a time
            for (int color = 0; color < 3; color++) {
                for (size_t i = 0; i < dst.datasize_ / 3; i++) {
                    dst.data_[color * (dst.datasize_ / 3) + i] = reinterpret_cast<uint8_t*>(data)[i * 3 + color];
                }
            }
        }
        else if (header.datatype == DT_RGBA32) {
            // Read a plane at a time, do not discard alpha
            for (int color = 0; color < 4; color++) {
                for (size_t i = 0; i < dst.datasize_ / 4; i++) {
                    dst.data_[color * (dst.datasize_ / 4) + i] = reinterpret_cast<uint8_t*>(data)[i * 4 + color];
                }
            }
        }
        else {
            is.read(reinterpret_cast<char*>(dst.data_), dst.datasize_);
        }

        ifile.close();

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
    else if (src.colortype_ == ColorType::GRAY || src.colortype_ == ColorType::none); // do nothing
    else {
        std::cerr << ECVL_WARNING_MSG << "Supported color types for NIfTI write are RGB, RGBA, GRAY and none." << endl;
        return false;
    }

    const Image& tmp1 = use_tmp1 ? tmp : src;
    bool use_tmp2 = false;

    string target_channel_model = "xyzt";
    string target_channel = "";
    size_t num_channels = tmp1.channels_.size();
    size_t c_pos = tmp1.channels_.find('c');
    if (c_pos != string::npos) {
        target_channel = "c";
        num_channels--;
    }
    for (int i = 0; i < num_channels && i < 4; i++) {
        target_channel += target_channel_model[i];
    }
    for (int i = 4; i < num_channels; i++) {
        target_channel += 'o';
    }
    RearrangeChannels(tmp1, tmp, target_channel);
    use_tmp2 = true;

    const Image& img = use_tmp2 ? tmp : tmp1;

    ofstream os;
    os.exceptions(ifstream::failbit | ifstream::eofbit | ifstream::badbit);

    try {
        os.open(filename, ios::binary);

        // Header fields
        ByteWriter wr(os);

        wr(348);

        wr.Fill(35);

        try {
            wr(any_cast<char>(img.GetMeta("dim_info").Get()));
        }
        catch(const exception&) {
            wr.Fill(1);
        }

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
        else if (img.colortype_ == ColorType::GRAY || img.colortype_ == ColorType::none) {
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

        try {
            wr(any_cast<char>(img.GetMeta("xyzt_units").Get()));

            wr.Fill(8);
            wr(any_cast<float>(img.GetMeta("slice_duration").Get()));
            wr(any_cast<float>(img.GetMeta("toffset").Get()));

            wr.Fill(112);   // We don't use those fields for now
            wr(any_cast<short>(img.GetMeta("qform_code").Get()));
            wr(any_cast<short>(img.GetMeta("sform_code").Get()));
            wr(any_cast<float>(img.GetMeta("quatern_b").Get()));
            wr(any_cast<float>(img.GetMeta("quatern_c").Get()));
            wr(any_cast<float>(img.GetMeta("quatern_d").Get()));
            wr(any_cast<float>(img.GetMeta("qoffset_x").Get()));
            wr(any_cast<float>(img.GetMeta("qoffset_y").Get()));
            wr(any_cast<float>(img.GetMeta("qoffset_z").Get()));
            wr(any_cast<array<float, 4>>(img.GetMeta("srow_x").Get()));
            wr(any_cast<array<float, 4>>(img.GetMeta("srow_y").Get()));
            wr(any_cast<array<float, 4>>(img.GetMeta("srow_z").Get()));
        }
        catch (const exception&) {
            wr.Fill(205);
        }

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