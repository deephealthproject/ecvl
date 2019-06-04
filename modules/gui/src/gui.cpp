#include "ecvl/gui.h"

namespace ecvl{

BEGIN_EVENT_TABLE(wxImagePanel, wxPanel)
EVT_PAINT(wxImagePanel::paintEvent)
EVT_SIZE(wxImagePanel::OnSize)
END_EVENT_TABLE()

void wxImagePanel::SetImage(const wxImage& img)
{
    image = img.Copy();
}

void wxImagePanel::paintEvent(wxPaintEvent & evt)
{
    wxPaintDC dc(this);
    render(dc);
    wxTheApp->OnExit();
}

void wxImagePanel::render(wxDC&  dc)
{
    int neww, newh;
    dc.GetSize(&neww, &newh);

    if (neww != w || newh != h)
    {
        resized = wxBitmap(image.Scale(neww, newh));
        w = neww;
        h = newh;
        dc.DrawBitmap(resized, 0, 0, false);
    }
    else {
        dc.DrawBitmap(resized, 0, 0, false);
    }
}

void wxImagePanel::OnSize(wxSizeEvent& event) 
{
    Refresh();
    event.Skip();
}

wxImage wx_from_mat(Image &img) 
{
    wxImage wx(img.dims_[0], img.dims_[1], (uint8_t*)malloc(img.datasize_), false);
    uint8_t* s = img.data_;
    uint8_t* d = wx.GetData();

    if (img.channels_ == "xyc") {
        //BGR to RGB
        uint8_t *tmp;
        auto i = img.Begin<uint8_t>();
        for (int p = 0; p < 3; ++p) {
            tmp = d + (2 - p);
            for (int r = 0; r < img.dims_[1]; r++) {
                for (int c = 0; c < img.dims_[0]; c++) {
                    *tmp = *i;
                    tmp += 3;
                    ++i;
                }
            }
        }
    }
    else {
        throw std::runtime_error("Not implemented");
    }
    return wx;
}

bool ShowApp::OnInit()
{
    wxInitAllImageHandlers();

    wxFrame *frame = new wxFrame(NULL, wxID_ANY, wxT("Image"), wxPoint(10, 10), wxSize(img_.dims_[0], img_.dims_[1]));
    wxImagePanel *drawPane = new wxImagePanel(frame);

    wxImage imwx1 = wx_from_mat(img_);
    drawPane->SetImage(imwx1);

    frame->Show();

    return true;
}

void ImShow(const Image& img)
{
    wxApp* App = new ShowApp(img);
    wxApp::SetInstance(App);

    wxEntry();
}

} // namespace ecvl