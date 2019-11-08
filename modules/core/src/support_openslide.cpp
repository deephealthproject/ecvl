#include "ecvl/core/support_openslide.h"

#include <iostream>
#include "openslide.h"

#include "ecvl/core/standard_errors.h"

using namespace std::filesystem;
using namespace std;

namespace ecvl
{

bool HamamatsuRead(const path &filename, Image &dst, int x, int y, int w, int h)
{
    bool return_value = true;

    openslide_t *osr = openslide_open(filename.string().c_str());

    if (osr == NULL || openslide_get_error(osr) != NULL)
    {
        cout << ECVL_ERROR_MSG << "Openslide cannot open " << filename << endl;
        return_value = false;
    }
    else
    {
        uint32_t *d = new uint32_t[4 * w * h];
        openslide_read_region(osr, d, x, y, 0, w, h);
        
        dst.Create({ 3, static_cast<int>(w), static_cast<int>(h) }, DataType::uint8, "cxy", ColorType::BGR);

        uint8_t a, r, g, b;
        uint32_t pixel;
        for (int i = 0, j = 0; i < dst.datasize_; ++j, ++i)
        {
            pixel = d[j];
            a = pixel >> 24;
            switch (a)
            {
            case 0:
                r = g = b = 0;
                break;
            case 255:
                r = (pixel >> 16) & 0xff;
                g = (pixel >> 8) & 0xff;
                b = pixel & 0xff;
                break;
            default:
                r = 255 * ((pixel >> 16) & 0xff) / a;
                g = 255 * ((pixel >> 8) & 0xff) / a;
                b = 255 * (pixel & 0xff) / a;
            }
            dst.data_[i] = b;
            dst.data_[++i] = g;
            dst.data_[++i] = r;
        }
    }

    openslide_close(osr);

    if (!return_value) {
        dst = Image();
    }

    return !dst.IsEmpty();
}

} // namespace ecvl