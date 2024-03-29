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

#include <glad/glad.h>

#include "ecvl/gui.h"

#include <ctime>

#include <iostream>
#include <algorithm>

#include <wx/glcanvas.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <opencv2/core.hpp>

#include "ecvl/gui/shader.h"
#include "ecvl/core/datatype_matrix.h"
#include "ecvl/core/standard_errors.h"

namespace ecvl
{
// settings

template <DataType DT> // src type
struct NormalizeToUint8Struct
{
    static void _(const Image& src, Image& dst)
    {
        dst.Create(src.dims_, DataType::uint8, src.channels_, src.colortype_, src.spacings_);

        ConstView<DT> src_v(src);
        View<DataType::uint8> dst_v(dst);

        // find max and min
        TypeInfo_t<DT> max = *std::max_element(src_v.Begin(), src_v.End());
        TypeInfo_t<DT> min = *std::min_element(src_v.Begin(), src_v.End());

        auto dst_it = dst_v.Begin();
        auto src_it = src_v.Begin();
        auto src_end = src_v.End();
        for (; src_it != src_end; ++src_it, ++dst_it) {
            (*dst_it) = (((*src_it) - min) * 255) / (max - min);
        }
    }
};

void NormalizeToUint8(const Image& src, Image& dst)
{
    Table1D<NormalizeToUint8Struct> table;
    table(src.elemtype_)(src, dst);
}

class BasicGLPane : public wxGLCanvas
{
    wxGLContext* m_context;
    wxTimer timer;
    Shader ourShader;
    unsigned int VBO3D, VAO3D, EBO, texture3D;
    const float radius = 0.7f;
    clock_t t;
    glm::mat4 view;
    glm::mat4 orientation;
    glm::mat4 ruota;
    bool enable_rotation;
    const float fps = 100;
    float period = 5;
    int slices = 500;

    std::string vertex_shader =
        "#version 330 core\n"
        "layout(location = 0) in vec2 xyPos;\n"
        "layout(location = 1) in vec2 aTexCoord;\n"

        "uniform mat4 model;\n"
        "uniform mat4 view;\n"
        "uniform mat4 projection;\n"

        "uniform float radius;\n"
        "uniform int slices;\n"

        "uniform mat3 ruota;\n"
        "uniform mat3 orientation;\n"
        "uniform mat3 scala;\n"

        "out vec3 TexCoord;\n"

        "void main()\n"
        "{"
        "    float zPos = -radius + (radius / slices) + gl_InstanceID * ((radius * 2) / slices);\n"  
        "    gl_Position = projection * view * vec4(xyPos, zPos, 1.0);"
        "    TexCoord = scala * orientation * ruota * vec3(aTexCoord, zPos) + vec3(0.5f, 0.5f, 0.5f);\n"    
        "}";

    std::string fragment_shader =
        "#version 330 core\n"
        "out vec4 FragColor;\n"
        "in vec3 TexCoord;\n"

        "uniform sampler3D ourTexture;\n"

        "void main()"
        "{"
        "    FragColor = texture(ourTexture, TexCoord);\n"
        "}";

public:
    BasicGLPane(wxFrame* parent, int* args, const Image& img);
    virtual ~BasicGLPane();

    void OnTimer(wxTimerEvent& event);

    int getWidth();
    int getHeight();

    void Render(wxPaintEvent& evt);

    void SetViewport();

    void KeyReleased(wxKeyEvent& evt);
    void MouseWheelMoved(wxMouseEvent& evt);

    // events
    //void mouseMoved(wxMouseEvent& event);
    //void mouseDown(wxMouseEvent& event);
    //void mouseWheelMoved(wxMouseEvent& event);
    //void mouseReleased(wxMouseEvent& event);
    //void rightClick(wxMouseEvent& event);
    //void mouseLeftWindow(wxMouseEvent& event);
    //void keyPressed(wxKeyEvent& event);
    //void keyReleased(wxKeyEvent& event);

    DECLARE_EVENT_TABLE()
};

BEGIN_EVENT_TABLE(BasicGLPane, wxGLCanvas)
//EVT_MOTION(BasicGLPane::mouseMoved)
//EVT_LEFT_DOWN(BasicGLPane::mouseDown)
//EVT_LEFT_UP(BasicGLPane::mouseReleased)
//EVT_RIGHT_DOWN(BasicGLPane::rightClick)
//EVT_LEAVE_WINDOW(BasicGLPane::mouseLeftWindow)
//EVT_KEY_DOWN(BasicGLPane::keyPressed)
EVT_KEY_UP(BasicGLPane::KeyReleased)
EVT_MOUSEWHEEL(BasicGLPane::MouseWheelMoved)
EVT_PAINT(BasicGLPane::Render)
EVT_TIMER(wxID_ANY, BasicGLPane::OnTimer)
END_EVENT_TABLE()

class Show3DApp : public wxApp
{
    const Image& img_;

    virtual bool OnInit();

    wxFrame* frame;
    BasicGLPane* glPane;
public:
    Show3DApp(const Image& img) : img_{ img } {}
};

bool Show3DApp::OnInit()
{
    wxBoxSizer* sizer = new wxBoxSizer(wxHORIZONTAL);
    wxFrame* frame = new wxFrame(NULL, wxID_ANY, wxT("3D Image"), wxPoint(10, 10), wxSize(500, 500));
    int args[] = { WX_GL_RGBA, WX_GL_DOUBLEBUFFER, WX_GL_DEPTH_SIZE, 16, 0 };

    glPane = new BasicGLPane(frame, args, img_);
    sizer->Add(glPane, 1, wxEXPAND);

    frame->SetSizer(sizer);
    frame->SetAutoLayout(true);

    frame->Show();
    return true;
}

void ImShow3D(const Image& img)
{
    if (img.channels_.find("xyz") == std::string::npos) {
        std::cout << "Image must have channels xyz" << std::endl;
        return;
    }
    wxApp* App = new Show3DApp(img);
    wxApp::SetInstance(App);

    int argc = 0;
    wxEntry(argc, static_cast<char**>(nullptr));
}

void BasicGLPane::Render(wxPaintEvent& evt)
{
    if (!IsShown()) return;

    wxPaintDC(this); // only to be used in paint events. use wxClientDC to paint outside the paint event

    SetViewport();

    if (enable_rotation) {
        ruota = glm::rotate(ruota, glm::radians(360.f / (period * fps)), glm::vec3(0.0f, 1.0f, 0.0f));
    }
    ourShader.setMat3("ruota", glm::mat3(ruota));

    // render
    // ------
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindVertexArray(VAO3D);

    glBindTexture(GL_TEXTURE_3D, texture3D);    

    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*) (sizeof(unsigned int) * 0), slices);

    SwapBuffers();
}

BasicGLPane::BasicGLPane(wxFrame* parent, int* args, const Image& src_img) :
    wxGLCanvas(parent, wxID_ANY, args, wxDefaultPosition, wxDefaultSize, wxFULL_REPAINT_ON_RESIZE), timer(this, wxID_ANY)
{
    m_context = new wxGLContext(this);

    SetCurrent(*m_context);

    if (!gladLoadGL()) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return;
    }

    Image uint8_conversion;
    NormalizeToUint8(src_img, uint8_conversion);
    const Image& img = uint8_conversion;
    //const Image& img = src_img;

    //std::cout << GLVersion.major << "." << GLVersion.minor << std::endl;

    timer.Start(1000 / fps);
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices3D[] = {
        // positions   // texture coords
         0.5f,  0.5f,  +radius, -radius,  // top right
         0.5f, -0.5f,  +radius, +radius,  // bottom right
        -0.5f, -0.5f,  -radius, +radius,  // bottom left
        -0.5f,  0.5f,  -radius, -radius   // top left
    };
    unsigned int indices3D[] = {
        0, 1, 2,
        0, 2, 3,
    };

    glGenVertexArrays(1, &VAO3D);
    glGenBuffers(1, &VBO3D);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO3D);

    glBindBuffer(GL_ARRAY_BUFFER, VBO3D);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices3D), vertices3D, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices3D), indices3D, GL_STATIC_DRAW);

    int width;
    int height;
    int depth;
    float dw = 1.f;
    float dh = 1.f;
    float dd = 1.f;

    if (!img.channels_.compare(0, 3, "xyz")) {
        width = img.dims_[0];
        height = img.dims_[1];
        depth = img.dims_[2];
        if (img.spacings_.size() >= 3) {
            dw = img.spacings_[0];
            dh = img.spacings_[1];
            dd = img.spacings_[2];
        }
    }
    else if (!img.channels_.compare(1, 3, "xyz")) {
        width = img.dims_[1];
        height = img.dims_[2];
        depth = img.dims_[3];
        if (img.spacings_.size() >= 4) {
            dw = img.spacings_[1];
            dh = img.spacings_[2];
            dd = img.spacings_[3];
        }
    }
    else {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    float scale_w = (1.f / width) / dw;
    float scale_h = (1.f / height) / dh;
    float scale_d = (1.f / depth) / dd;

    float scale_min = std::min(std::min(scale_w, scale_h), scale_d);
    float coeff = 1 / scale_min;
    scale_w *= coeff;
    scale_h *= coeff;
    scale_d *= coeff;

    const glm::mat4 scala = glm::scale(glm::mat4(1.f), glm::vec3(scale_w, scale_h, scale_d));

    // Going 3D
    glEnable(GL_TEXTURE_3D);

    glGenTextures(1, &texture3D);
    glBindTexture(GL_TEXTURE_3D, texture3D);

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Convert the texture data to RGBA
    unsigned char black_threshold = 30;
    unsigned char alpha = 100;
    unsigned char* data;

    if (img.colortype_ == ColorType::GRAY || img.colortype_ == ColorType::none) {
        if (!img.channels_.compare(0, 3, "xyz")) {
            data = new unsigned char[img.dims_[0] * img.dims_[1] * img.dims_[2] * 4];
            for (int i = 0; i < img.dims_[0] * img.dims_[1] * img.dims_[2]; i++) {
                data[i * 4] = img.data_[i];
                data[i * 4 + 1] = img.data_[i];
                data[i * 4 + 2] = img.data_[i];
                data[i * 4 + 3] = img.data_[i];
                //data[i * 4 + 3] = alpha;
                if (data[i * 4 + 0] < black_threshold) {
                    data[i * 4 + 3] = 0;
                }
            }
        }
        else {
            ECVL_ERROR_NOT_IMPLEMENTED
        }
    }
    else if (img.colortype_ == ColorType::RGB) {
        if (!img.channels_.compare(0, 3, "xyz")) {
            data = new unsigned char[img.dims_[0] * img.dims_[1] * img.dims_[2] * 4];
            for (int i = 0; i < img.dims_[0] * img.dims_[1] * img.dims_[2]; i++) {
                //memcpy(data + i * 4, img.data_ + i * 3, 3);
                data[i * 4 + 0] = img.data_[img.strides_.back() * 0 + i];
                data[i * 4 + 1] = img.data_[img.strides_.back() * 1 + i];
                data[i * 4 + 2] = img.data_[img.strides_.back() * 2 + i];
                if (data[i * 4 + 0] < black_threshold && data[i * 4 + 1] < black_threshold && data[i * 4 + 2] < black_threshold) {
                    data[i * 4 + 3] = 0;
                }
                else {
                    //data[i * 4 + 3] = data[i * 4];
                    data[i * 4 + 3] = alpha;
                }
            }
        }
        else {
            ECVL_ERROR_NOT_IMPLEMENTED
        }
    }
    else if (img.colortype_ == ColorType::BGR) {
        if (!img.channels_.compare(0, 3, "xyz")) {
            data = new unsigned char[img.dims_[0] * img.dims_[1] * img.dims_[2] * 4];
            for (int i = 0; i < img.dims_[0] * img.dims_[1] * img.dims_[2]; i++) {
                //memcpy(data + i * 4, img.data_ + i * 3, 3);
                data[i * 4 + 0] = img.data_[img.strides_.back() * 2 + i];
                data[i * 4 + 1] = img.data_[img.strides_.back() * 1 + i];
                data[i * 4 + 2] = img.data_[img.strides_.back() * 0 + i];
                if (data[i * 4 + 0] < black_threshold && data[i * 4 + 1] < black_threshold && data[i * 4 + 2] < black_threshold) {
                    data[i * 4 + 3] = 0;
                }
                else {
                    //data[i * 4 + 3] = data[i * 4];
                    data[i * 4 + 3] = alpha;
                }
            }
        }
        else if (!img.channels_.compare(1, 3, "xyz")) {
            data = new unsigned char[img.dims_[1] * img.dims_[2] * img.dims_[3] * 4];
            for (int i = 0; i < img.dims_[1] * img.dims_[2] * img.dims_[3]; i++) {
                data[i * 4 + 0] = img.data_[i * 3 + 2];
                data[i * 4 + 1] = img.data_[i * 3 + 1];
                data[i * 4 + 2] = img.data_[i * 3 + 0];
                if (data[i * 4 + 0] < black_threshold && data[i * 4 + 1] < black_threshold && data[i * 4 + 2] < black_threshold) {
                    data[i * 4 + 3] = 0;
                }
                else {
                    //data[i * 4 + 3] = data[i * 4];
                    data[i * 4 + 3] = alpha;
                }
            }
        }
        else {
            ECVL_ERROR_NOT_IMPLEMENTED
        }
    }
    else {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, width, height, depth, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    delete[] data;

    glGenerateMipmap(GL_TEXTURE_3D);

    // Transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // render container
    ourShader.init(vertex_shader, fragment_shader);
    ourShader.use();

    ourShader.setFloat("radius", radius);
    ourShader.setInt("slices", slices);

    orientation = glm::mat4(1.f);
    ourShader.setMat3("orientation", glm::mat3(orientation));

    ruota = glm::mat4(1.f);
    enable_rotation = true;

    view = glm::mat4(1.0f);
    view = glm::translate(view, glm::vec3(0.0f, 0.0f, -2.f));
    //view = glm::rotate(view, glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f));
    ourShader.setMat4("view", view);

    glm::mat4 projection;
    projection = glm::perspective(glm::radians(35.f), (float)GetSize().x / (float)GetSize().y, 0.1f, 100.0f);  // projective
    //projection = glm::ortho(-1.f, 1.f, -1.f, 1.f);                                                          // orthographic
    ourShader.setMat4("projection", projection);

    ourShader.setMat3("scala", glm::mat3(scala));

    // To avoid flashing on MSW
    SetBackgroundStyle(wxBG_STYLE_CUSTOM);
}

BasicGLPane::~BasicGLPane()
{
    glDeleteVertexArrays(1, &VAO3D);
    glDeleteBuffers(1, &VBO3D);
    glDeleteBuffers(1, &EBO);

    delete m_context;
}

void BasicGLPane::OnTimer(wxTimerEvent& event)
{
    // do whatever you want to do every second here
    Refresh();
    Update();
}

void BasicGLPane::MouseWheelMoved(wxMouseEvent& evt)
{
    int mouse_rotation = evt.GetWheelRotation();
    //view = glm::translate(view, glm::vec3(0.0f, 0.0f, (float)mouse_rotation / 1000));
    //ourShader.setMat4("view", view);
    slices += mouse_rotation / 10;
    if (slices <= 1)
        slices = 1;
    ourShader.setInt("slices", slices);
}

void BasicGLPane::KeyReleased(wxKeyEvent& evt)
{
    int key_code = evt.GetKeyCode();
    if (key_code == WXK_ESCAPE) {
        // Close window
    }
    else if (key_code == WXK_SPACE) {
        enable_rotation = !enable_rotation;
    }
    else if (key_code == WXK_UP) {
        orientation = glm::rotate(orientation, glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f));
        ourShader.setMat3("orientation", glm::mat3(orientation));
    }
    else if (key_code == WXK_DOWN) {
        orientation = glm::rotate(orientation, glm::radians(-90.f), glm::vec3(1.f, 0.f, 0.f));
        ourShader.setMat3("orientation", glm::mat3(orientation));
    }
    else if (key_code == WXK_RIGHT) {
        orientation = glm::rotate(orientation, glm::radians(90.f), glm::vec3(0.f, 0.f, 1.f));
        ourShader.setMat3("orientation", glm::mat3(orientation));
    }
    else if (key_code == WXK_LEFT) {
        orientation = glm::rotate(orientation, glm::radians(-90.f), glm::vec3(0.f, 0.f, 1.f));
        ourShader.setMat3("orientation", glm::mat3(orientation));
    }
}

void BasicGLPane::SetViewport()
{
    wxSize s = GetSize();
    int min = std::min(s.x, s.y);
    glViewport((s.x - min) / 2, (s.y - min) / 2, min, min);
}
} // namespace ecvl