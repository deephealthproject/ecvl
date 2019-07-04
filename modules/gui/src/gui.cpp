#include "ecvl/gui.h"

#include "ecvl/core/imgproc.h"

namespace ecvl{

   

    /** @brief wxImagePanel creates a wxPanel to contain an Image.

    */
    class wxImagePanel : public wxPanel
    {
        wxImage wxImage_;
        Image ecvlImage_;
        wxBitmap resized_;
        int w_, h_;
        void PaintEvent(wxPaintEvent & evt);
        void Render(wxDC& dc);

    public:
        wxImagePanel(wxPanel* parent) : wxPanel(parent) {}
        wxImagePanel(wxFrame* parent) : wxPanel(parent) {}
        wxImagePanel(wxNotebook* parent, wxWindowID id) : wxPanel(parent, id) {}
        void SetImage(const wxImage& img);
        wxImage GetWXImage();
        Image GetECVLImage();
        void OnSize(wxSizeEvent& event);
        bool IsEmpty();

        DECLARE_EVENT_TABLE()
    };

    /** @brief ShowApp is a custom wxApp which allows you to visualize an ECVL Image.

*/
    class ShowApp : public wxApp
    {
        Image img_;      /**< Image to be shown. */

    public:
        /** @brief Initialization function. Starts the main loop of the application.

            The OnInit() function creates a wxFrame which has the width and the height of the Image that has to be shown.
            It also creates the wxImagePanel which contains the frame and employs the conversion from Image to
            wxImage.
            It set the wxImage in the frame and starts the main loop of the ShowApp.
        */
        bool OnInit();

        /** @brief Constructor.

            The constructor creates a ShowApp initializing its Image with the given input Image.
        */
        ShowApp(const Image &img) : img_{ img } {}

    };

BEGIN_EVENT_TABLE(wxImagePanel, wxPanel)
EVT_PAINT(wxImagePanel::PaintEvent)
EVT_SIZE(wxImagePanel::OnSize)
END_EVENT_TABLE()

void wxImagePanel::SetImage(const wxImage& img)
{
    wxImage_ = img.Copy();
    ecvlImage_ = ImgFromWx(wxImage_);
}

wxImage wxImagePanel::GetWXImage()
{
    return wxImage_;
}

Image wxImagePanel::GetECVLImage()
{
    return ecvlImage_;
}

bool wxImagePanel::IsEmpty()
{
    if (wxImage_.IsOk())
        return false;
    else return true;
}

void wxImagePanel::PaintEvent(wxPaintEvent & evt)
{
    wxPaintDC dc(this);
    Render(dc);
    wxTheApp->OnExit();
}

void wxImagePanel::Render(wxDC&  dc)
{
    int neww, newh;
    dc.GetSize(&neww, &newh);
    resized_ = wxBitmap(wxImage_.Scale(neww, newh));
    w_ = neww;
    h_ = newh;
    dc.DrawBitmap(resized_, 0, 0, false);
}

void wxImagePanel::OnSize(wxSizeEvent& event)
{
    Refresh();
    event.Skip();
}

bool ShowApp::OnInit()
{
    wxInitAllImageHandlers();
    wxImage imwx = WxFromImg(img_);

    wxBoxSizer* sizer = new wxBoxSizer(wxHORIZONTAL);
    wxFrame *frame = new wxFrame(NULL, wxID_ANY, wxT("Image"), wxPoint(10, 10), wxSize(imwx.GetWidth(), imwx.GetHeight()));
    wxImagePanel *drawPane = new wxImagePanel(frame);

    drawPane->SetImage(imwx);
    sizer->Add(drawPane, 1, wxEXPAND);

    frame->SetSizer(sizer);
    frame->SetAutoLayout(true);

    frame->Show();

    return true;
}

void ImShow(const Image& img)
{
    wxApp* App = new ShowApp(img);
    wxApp::SetInstance(App);

    int argc = 0;
    wxEntry(argc, static_cast<char**>(nullptr));
}

wxImage WxFromImg(Image& img)
{
    if (img.colortype_ != ColorType::RGB)
        ChangeColorSpace(img, img, ColorType::RGB);

    if (img.channels_ != "cxy")
        RearrangeChannels(img, img, "cxy");

    wxImage wx(img.dims_[1], img.dims_[2], (uint8_t*)malloc(img.datasize_), false);
    uint8_t* wx_data = wx.GetData();

    if(img.contiguous_)
        memcpy(wx_data, img.data_, img.datasize_);
    else
    {
        auto i = img.Begin<uint8_t>(), e = img.End<uint8_t>();
        for (; i != e; ++i) {
            *wx_data = *i;
            ++wx_data;
        }
    }

    return wx;
}

Image ImgFromWx(const wxImage& wx)
{
    Image img({ 3, wx.GetWidth(), wx.GetHeight() }, DataType::uint8, "cxy", ColorType::RGB);
    uint8_t* wx_data = wx.GetData();

    if (img.contiguous_)
        memcpy(img.data_, wx_data, img.datasize_);
    else
    {
        auto i = img.Begin<uint8_t>(), e = img.End<uint8_t>();
        for (; i != e; ++i) {
            *i = *wx_data;
            ++wx_data;
        }
    }

    return img;
}

} // namespace ecvl