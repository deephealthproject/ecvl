#include "ecvl/core/support_nifti.h"

#include <string>

#include <nifti/nifti1_io.h>

using namespace std;

namespace ecvl {

    bool NiftiRead(const std::string& filename, Image& dst) {

        nifti_image* nim = NULL;

        try {

            znzFile f = nifti_image_open(filename.c_str(), "r", &nim);
            if (f == NULL || nim == NULL) {
                throw std::runtime_error("Can't open nifti image.\n");
            }

            if (nifti_image_load(nim) != 0) {
                throw std::runtime_error("Can't load nifti pixel data.\n");
            }

            // We only support images and volumes by now, not temporal series or different stuff
            if (nim->ndim < 2 || nim->ndim > 3) {
                throw std::runtime_error("Unsupported number of dimensions.\n");
            }

            // Convert nifti_image into ecvl::Image
            std::vector<int> dims;
            for (int i = 0; i < nim->ndim; i++) {
                dims.push_back(nim->dim[i + 1]);
            }

            DataType data_type;
            switch (nim->datatype) {

            case  DT_BINARY:           data_type = DataType::uint8;     break;               /* binary (1 bit/voxel)         */                      // qua bisogna leggere bit a bit D:
            case  DT_UNSIGNED_CHAR:    data_type = DataType::uint8;     break;                      /* unsigned char (8 bits/voxel) */
            case  DT_SIGNED_SHORT:     data_type = DataType::int16;     break;                     /* signed short (16 bits/voxel) */
            case  DT_SIGNED_INT:       data_type = DataType::int32;     break;             /* signed int (32 bits/voxel)   */
            case  DT_FLOAT:            data_type = DataType::float32;   break;                /* float (32 bits/voxel)        */
            case  DT_DOUBLE:           data_type = DataType::float64;   break;                 /* double (64 bits/voxel)       */
            case  DT_RGB:              data_type = DataType::uint8;     break;            /* RGB triple (24 bits/voxel)   */                          // attenzione perché sono 3 canali
            case  DT_INT8:             data_type = DataType::int8;      break;            /* signed char (8 bits)         */
            case  DT_UINT16:           data_type = DataType::uint16;    break;                /* unsigned short (16 bits)     */
            case  DT_UINT32:           data_type = DataType::uint32;    break;                /* unsigned int (32 bits)       */
            case  DT_INT64:            data_type = DataType::int64;     break;              /* long long (64 bits)          */
            case  DT_UINT64:           data_type = DataType::uint64;    break;                /* unsigned long long (64 bits) */
            case  DT_RGBA32:           data_type = DataType::uint8;     break;               /* 4 byte RGBA (32 bits/voxel)  */                       // attenzione perché sono 4 canali
            // case  DT_COMPLEX256:       data_type = DataType::none;      break;                  /* long double pair (256 bits)  */                    // non supportato (che cos'è un complex?)
            // case  DT_COMPLEX128:       data_type = DataType::none;      break;                  /* double pair (128 bits)       */                    // non supportato (che cos'è un complex?)
            // case  DT_FLOAT128:         data_type = DataType::none;      break;                /* long double (128 bits)       */                      // non supportato
            // case  DT_UNKNOWN:          data_type = DataType::none;      break;               /* what it says, dude           */
            // case  DT_ALL:              data_type = DataType::none;      break;           /* not very useful (?)          */
            // case  DT_COMPLEX:          data_type = DataType::none;      break;               /* complex (64 bits/voxel)      */                       // che cos'è un complex?
            default:                   throw runtime_error("Unsupported data type.\n");

            }

            ColorType color_type = ColorType::none;

            if (data_type != DataType::none) {
                switch (nim->datatype) {

                case DT_RGB:                    color_type = ColorType::RGB;    break;
                case DT_RGBA32:                 color_type = ColorType::RGB;    break;      // This should be RGBA but we don't have it!
                default:                        color_type = ColorType::GRAY;   break;

                }

            }

            string channels;

            switch (color_type) {

            case ColorType::RGB:     dims.push_back(3);     channels += 'c';    break;
            case ColorType::GRAY:    dims.push_back(1);     channels += 'c';    break;
            default:                                                            break;

            }

            dst.Create(dims, data_type, channels, color_type);

            // Copia i pixel
            if (nim->datatype == DT_BINARY) {
                // leggi bit a bit
            }
            else if (nim->datatype == DT_RGB) {
                // leggi un piano alla volta
            }
            else if (nim->datatype == DT_RGBA32) {
                // leggi un piano alla volta, scartando alpha
            }
            else {

                memcpy(dst.data_, nim->data, dst.datasize_);

            }

            znzclose(f);
            nifti_image_free(nim);
        }
        catch (const std::runtime_error&) {
            dst = Image();
            return false;
        }
    }


} // namespace evl