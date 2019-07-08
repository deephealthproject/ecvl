#include <glad/glad.h>

#include "ecvl/gui.h"

#include <ctime>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <opencv2/core.hpp>

#include "ecvl/gui/shader.h"

namespace ecvl {
    // settings
    const unsigned int SCR_WIDTH = 800;
    const unsigned int SCR_HEIGHT = 800;

    class BasicGLPane : public wxGLCanvas
    {
        wxGLContext *m_context;
        wxTimer timer;
        Shader ourShader;
        unsigned int VBO3D, VAO3D, EBO, texture3D;
        const float radius = 0.7f;
        clock_t t;
        std::string vertex_shader =
            "#version 330 core\n"
            "layout(location = 0) in vec2 xyPos;\n"
            "layout(location = 1) in vec3 aTexCoord;\n"

            "uniform float zPos;\n"
            "uniform mat4 model;\n"
            "uniform mat4 projection;\n"
            "uniform mat4 view;\n"

            "out vec2 TexCoord;\n"

            "void main()\n"
            "{"
            "    gl_Position = projection * view * vec4(xyPos, zPos, 1.0);"
            "    TexCoord = aTexCoord.xy;"
            "}";

        std::string fragment_shader =
            "#version 330 core\n"
            "out vec4 FragColor;\n"

            "in vec2 TexCoord;\n"

            "uniform sampler3D ourTexture;\n"
            "uniform float zPos;\n"
            "uniform mat4 ruota;\n"
            "uniform float radius;\n"
            "uniform mat4 scala;\n"

            "void main()"
            "{"
            "    FragColor = texture(ourTexture, (scala * ruota * vec4(vec3(TexCoord, zPos), 1)).xyz + vec3(0.5f, 0.5f, 0.5f));"
            "}";

    public:
        BasicGLPane(wxFrame* parent, int* args, const Image &img);
        virtual ~BasicGLPane();
 
        void OnTimer(wxTimerEvent& event);

        int getWidth();
        int getHeight();

        void Render(wxPaintEvent& evt);

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
        //EVT_KEY_UP(BasicGLPane::keyReleased)
        //EVT_MOUSEWHEEL(BasicGLPane::mouseWheelMoved)
        EVT_PAINT(BasicGLPane::Render)
        EVT_TIMER(wxID_ANY, BasicGLPane::OnTimer)
        END_EVENT_TABLE()

        class Show3DApp : public wxApp
    {
        Image img_;

        virtual bool OnInit();

        wxFrame *frame;
        BasicGLPane * glPane;
    public:
        Show3DApp(const Image &img) : img_{ img } {}

    };

    bool Show3DApp::OnInit()
    {
        wxBoxSizer* sizer = new wxBoxSizer(wxHORIZONTAL);
        wxFrame *frame = new wxFrame(NULL, wxID_ANY, wxT("3D Image"), wxPoint(10, 10), wxSize(500, 500));
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
        if (img.channels_.find("xyz") == std::string::npos)
        {
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
       
        glViewport(0, 0, GetSize().x, GetSize().y);
        glm::mat4 ruota(1.0f);
        ruota = glm::rotate(ruota, ((float)clock() / CLOCKS_PER_SEC), glm::vec3(0.0f, 1.0f, 0.0f));
        ourShader.setMat4("ruota", ruota);

        glm::mat4 trasla = glm::mat4(1.0f);

        // render
        // ------
        glClearColor(0.f, 0.f, 0.f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBindVertexArray(VAO3D);

        glEnable(GL_TEXTURE_3D);
        glBindTexture(GL_TEXTURE_3D, texture3D);

        const int slices = 200;

        trasla = glm::mat4(1.f);

        for (int i = 0; i < slices; i++) {

            float z = -radius + (radius / slices) + i * ((radius * 2) / slices); // Z-coordinate of the quad slice (and of the texture slice)

            ourShader.setFloat("zPos", z);

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        }

        SwapBuffers();
    }

    BasicGLPane::BasicGLPane(wxFrame* parent, int* args, const Image &img) :
        wxGLCanvas(parent, wxID_ANY, args, wxDefaultPosition, wxDefaultSize, wxFULL_REPAINT_ON_RESIZE), timer(this, wxID_ANY)
    {
        m_context = new wxGLContext(this);

        SetCurrent(*m_context);

        if (!gladLoadGL())
            std::cout << "Failed to initialize GLAD" << std::endl;

        std::cout << GLVersion.major << "." << GLVersion.minor << std::endl;
        timer.Start(40);
        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------
        float vertices3D[] = {
            // positions         // texture coords
             0.5f,  0.5f, /*0.0f,*/  +radius, -radius,  // top right
             0.5f, -0.5f, /*0.0f,*/  +radius, +radius,  // bottom right
            -0.5f, -0.5f, /*0.0f,*/  -radius, +radius,  // bottom left
            -0.5f,  0.5f, /*0.0f,*/  -radius, -radius   // top left 
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
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices3D), indices3D, GL_STATIC_DRAW);

        int depth = img.dims_[2];
        int width = img.dims_[0];
        int height = img.dims_[1];

        glm::mat4 scala = glm::scale(glm::mat4(1.f), glm::vec3((float)width / width, (float)width / height, (float)width / depth));

        // Going 3D
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
        unsigned char* data = new unsigned char[img.dims_[0] * img.dims_[1] * img.dims_[2] * 4];
        
        if (img.colortype_ == ColorType::RGB)
        {
            for (int i = 0; i < img.dims_[0] * img.dims_[1] * img.dims_[2]; i++) {
                memcpy(data + i * 4, img.data_ + i * 3, 3);
                if (data[i * 4 + 0] < black_threshold && data[i * 4 + 1] < black_threshold && data[i * 4 + 2] < black_threshold) {
                    data[i * 4 + 3] = 0;
                }
                else {
                    data[i * 4 + 3] = data[i * 4];
                    //data[i * 4 + 3] = 100;
                }
            }
        }
        else if (img.colortype_ == ColorType::GRAY)
        {
            for (int i = 0; i < img.dims_[0] * img.dims_[1] * img.dims_[2]; i++) {
                data[i * 4] = img.data_[i];
                data[i * 4 + 1] = img.data_[i];
                data[i * 4 + 2] = img.data_[i];
                data[i * 4 + 3] = img.data_[i];
                if (data[i * 4 + 0] < black_threshold) {
                    data[i * 4 + 3] = 0;
                }
            }
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

        glm::mat4 view = glm::mat4(1.0f);
        view = glm::translate(view, glm::vec3(0.0f, 0.0f, -1.5f));
        //view = glm::rotate(view, glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f));
        ourShader.setMat4("view", view);

        glm::mat4 projection;
        projection = glm::perspective(glm::radians(35.f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);  // projective
        //projection = glm::ortho(-1.f, 1.f, -1.f, 1.f);                                                          // orthographic
        ourShader.setMat4("projection", projection);

        ourShader.setFloat("radius", radius);
        ourShader.setMat4("scala", scala);

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
} // namespace ecvl