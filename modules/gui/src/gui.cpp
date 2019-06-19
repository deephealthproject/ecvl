#include "ecvl/gui.h"

#include "ecvl/core/imgproc.h"

namespace ecvl{

BEGIN_EVENT_TABLE(wxImagePanel, wxPanel)
EVT_PAINT(wxImagePanel::PaintEvent)
EVT_SIZE(wxImagePanel::OnSize)
END_EVENT_TABLE()

void wxImagePanel::SetImage(const wxImage& img)
{
    image_ = img.Copy();
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

    if (neww != w_ || newh != h_)
    {
        resized_ = wxBitmap(image_.Scale(neww, newh));
        w_ = neww;
        h_ = newh;
        dc.DrawBitmap(resized_, 0, 0, false);
    }
    else {
        dc.DrawBitmap(resized_, 0, 0, false);
    }
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

    wxFrame *frame = new wxFrame(NULL, wxID_ANY, wxT("Image"), wxPoint(10, 10), wxSize(imwx.GetWidth(), imwx.GetHeight()));
    wxImagePanel *drawPane = new wxImagePanel(frame);

    drawPane->SetImage(imwx);

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
    auto img_data = img.Begin<uint8_t>();

    for (int j = 0; j < img.datasize_; j++) {
        *wx_data = *img_data;
        ++wx_data;
        ++img_data;
    }

    return wx;
}

Image ImgFromWx(const wxImage& wx)
{
    Image img({ 3, wx.GetWidth(), wx.GetHeight() }, DataType::uint8, "cxy", ColorType::RGB);
    uint8_t* wx_data = wx.GetData();
    auto img_data = img.Begin<uint8_t>();

    for (int j = 0; j < img.datasize_; j++) {
        *img_data = *wx_data;
        ++wx_data;
        ++img_data;
    }

    return img;
}

} // namespace ecvl