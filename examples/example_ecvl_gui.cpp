#include "ecvl/core.h"
#include "ecvl/gui.h"

using namespace ecvl;
using namespace std;

int main()
{
    // Open an Image
    Image img;
    if (!ImRead("../data/test.jpg", img)) {
        return EXIT_FAILURE;
    }
    ImWrite("img_orig.jpg", img);

    // Display an Image in a window
    ImShow(img);
    
    // Create a black Image 1000x1000
    Image black({ 1000, 1000, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
    auto i = black.Begin<uint8_t>(), e = black.End<uint8_t>();
    for (; i != e; ++i) {
        *i = 0;
    }

    // Convert Image in wxImage
    cout << "Executing WxFromImg" << endl;
    wxImage wx_image = WxFromImg(img);
    wxImage wx_black = WxFromImg(black);

    // Apply wxWidgets functions
    cout << "Executing wxWidgets functions" << endl;
    wx_black.Paste(wx_image, 100, 100);

    // Convert wxImage in Image
    cout << "Executing ImgFromWx" << endl;
    img = ImgFromWx(wx_black);
    ImShow(img);

    // Save an Image
    cout << "Executing ChangeColorSpace" << endl;
    ChangeColorSpace(img, img, ColorType::BGR);
    ImWrite("img_from_wx.jpg", img);

    return EXIT_SUCCESS;
}