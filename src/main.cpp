// src/main.cpp
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <vector>
#define GLAD_GL_IMPLEMENTATION
#include "glad/glad.h"
#include <GLFW/glfw3.h>
#define IMGUI_ENABLE_DOCKING
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"
#define CGLM_CLIPSPACE_ZERO_TO_ONE
#include "cglm/cglm.h"

#define XNN_IMPLEMENTATION
#include "../xnn.h"

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __linux__
    glfwWindowHintString(GLFW_X11_CLASS_NAME, "xnn");
    glfwWindowHintString(GLFW_X11_INSTANCE_NAME, "xnn");
#endif
    GLFWwindow* window = glfwCreateWindow(1600, 1000, "xnn", nullptr, nullptr);
    if (!window) return -1;

    GLFWmonitor* primary = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(primary);
    glfwSetWindowMonitor(window, primary, 0, 0, mode->width, mode->height, mode->refreshRate);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigWindowsMoveFromTitleBarOnly = true;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // === 3D Cube Setup ===
    float vertices[] = {
        -0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
         0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
         0.5f,  0.5f, -0.5f, 0.0f, 0.0f, 1.0f,
        -0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f, 1.0f, 0.0f, 1.0f,
         0.5f, -0.5f,  0.5f, 0.0f, 1.0f, 1.0f,
         0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f, 0.5f, 0.5f, 0.5f,
    };
    unsigned int indices[] = { 0,1,2,2,3,0, 4,5,6,6,7,4, 0,1,5,5,4,0, 3,2,6,6,7,3, 0,3,7,7,4,0, 1,2,6,6,5,1 };
    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);

    GLuint shader = glCreateProgram();
    const char* vs = "#version 130\nin vec3 aPos;in vec3 aColor;out vec3 color;uniform mat4 mvp;void main(){gl_Position=mvp*vec4(aPos,1);color=aColor;}";
    const char* fs = "#version 130\nin vec3 color;out vec4 fragColor;void main(){fragColor=vec4(color,1);}";
    GLuint vsh = glCreateShader(GL_VERTEX_SHADER); glShaderSource(vsh,1,&vs,nullptr); glCompileShader(vsh);
    GLuint fsh = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(fsh,1,&fs,nullptr); glCompileShader(fsh);
    glAttachShader(shader,vsh); glAttachShader(shader,fsh); glLinkProgram(shader);
    glDeleteShader(vsh); glDeleteShader(fsh);

    GLuint cubeFBO, cubeTexture, depthRB;
    glGenFramebuffers(1, &cubeFBO);
    glGenTextures(1, &cubeTexture);
    glGenRenderbuffers(1, &depthRB);
    glBindTexture(GL_TEXTURE_2D, cubeTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1024, 1024, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRB);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1024, 1024);
    glBindFramebuffer(GL_FRAMEBUFFER, cubeFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, cubeTexture, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRB);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    bool toggle_pressed = false;
    int saved_x = 100, saved_y = 100, saved_w = 1600, saved_h = 1000;
    bool show_cube_window = true;
    bool show_visualizer = true;

    // Neural network configuration
    static std::vector<int> layer_sizes = {784, 128, 10};
    static std::vector<int> layer_acts = {ACT_RELU, ACT_RELU, ACT_SOFTMAX};
    static int loss_type = LOSS_CE;
    static bool locked = false;
    static Network* net = nullptr;

    XNN_INIT();  // seed RNG

    const char* act_names = "Sigmoid\0Tanh\0ReLU\0Softmax\0\0";

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        bool toggle_now = glfwGetKey(window, GLFW_KEY_F11) == GLFW_PRESS ||
                         (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS && glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS);
        if (toggle_now && !toggle_pressed) {
            toggle_pressed = true;
            bool is_fullscreen = glfwGetWindowMonitor(window) != nullptr;
            if (is_fullscreen) {
                glfwGetWindowPos(window, &saved_x, &saved_y);
                glfwGetWindowSize(window, &saved_w, &saved_h);
                glfwSetWindowMonitor(window, nullptr, saved_x, saved_y, saved_w, saved_h, 0);
            } else {
                glfwSetWindowMonitor(window, primary, 0, 0, mode->width, mode->height, mode->refreshRate);
            }
        }
        if (!toggle_now) toggle_pressed = false;

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("Main DockSpace", nullptr, ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus);
        ImGui::PopStyleVar(3);
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

        ImGui::Begin("xnn Control Panel");
        ImGui::Text("xnn – Interactive Neural Network Editor");
        ImGui::Text("FPS: %.1f", io.Framerate);
        ImGui::Text("F11 or Alt+Enter: Toggle Fullscreen");
        ImGui::Separator();

        if (locked) {
            ImGui::Text("Configuration Locked – Network Allocated");
            if (ImGui::Button("Unlock & Free Memory")) {
                if (net) network_free(net);
                net = nullptr;
                locked = false;
            }
        } else {
            ImGui::Text("Edit network – Lock when ready");
            ImGui::InputInt("Input neurons", &layer_sizes[0]);
            layer_sizes[0] = std::max(1, layer_sizes[0]);
            ImGui::Combo("Input act", &layer_acts[0], act_names);

            for (size_t i = 1; i < layer_sizes.size() - 1; ++i) {
                char buf[64];
                sprintf(buf, "Hidden %zu neurons", i);
                ImGui::InputInt(buf, &layer_sizes[i]);
                layer_sizes[i] = std::max(1, layer_sizes[i]);
                sprintf(buf, "Hidden %zu act", i);
                ImGui::Combo(buf, &layer_acts[i], act_names);
            }

            if (ImGui::Button("Add Hidden Layer")) {
                layer_sizes.insert(layer_sizes.end() - 1, 128);
                layer_acts.insert(layer_acts.end() - 1, ACT_RELU);
            }
            ImGui::SameLine();
            if (layer_sizes.size() > 2 && ImGui::Button("Remove Last Hidden")) {
                layer_sizes.erase(layer_sizes.end() - 2);
                layer_acts.erase(layer_acts.end() - 2);
            }

            ImGui::InputInt("Output neurons", &layer_sizes.back());
            layer_sizes.back() = std::max(1, layer_sizes.back());
            ImGui::Combo("Output act", &layer_acts.back(), act_names);
            ImGui::Combo("Loss", &loss_type, "MSE\0Cross-Entropy\0\0");

            if (ImGui::Button("Lock Configuration & Allocate")) {
                std::vector<size_t> arch(layer_sizes.begin(), layer_sizes.end());
                net = network_alloc(arch.data(), arch.size(), layer_acts.data(), loss_type);
                locked = true;
            }
        }
        ImGui::End();

        // 3D Cube Window
        ImGui::Begin("3D Cube Viewport", &show_cube_window, ImGuiWindowFlags_NoScrollbar);
        ImVec2 vp_size = ImGui::GetContentRegionAvail();
        if (vp_size.x > 0 && vp_size.y > 0) {
            int w = (int)vp_size.x, h = (int)vp_size.y;
            static int last_w = 0, last_h = 0;
            if (w != last_w || h != last_h) {
                glBindTexture(GL_TEXTURE_2D, cubeTexture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
                glBindRenderbuffer(GL_RENDERBUFFER, depthRB);
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
                last_w = w; last_h = h;
            }
            glBindFramebuffer(GL_FRAMEBUFFER, cubeFBO);
            glViewport(0, 0, w, h);
            glEnable(GL_DEPTH_TEST);
            glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glUseProgram(shader);
            mat4 model{}, view{}, proj{}, mvp{};
            glm_mat4_identity(model);
            glm_rotate(model, (float)glfwGetTime() * 0.8f, (vec3){0.5f, 1.0f, 0.0f});
            glm_lookat((vec3){2,2,2}, (vec3){0,0,0}, (vec3){0,1,0}, view);
            glm_perspective(glm_rad(45.0f), (float)w/h, 0.1f, 100.0f, proj);
            glm_mat4_mul(proj, view, mvp);
            glm_mat4_mul(mvp, model, mvp);
            glUniformMatrix4fv(glGetUniformLocation(shader, "mvp"), 1, GL_FALSE, (float*)mvp);
            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
        ImGui::Image((void*)(intptr_t)cubeTexture, vp_size, ImVec2(0,1), ImVec2(1,0));
        ImGui::End();

        // Network Visualizer
        ImGui::Begin("Network Visualizer", &show_visualizer);
        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
        ImVec2 canvas_size = ImGui::GetContentRegionAvail();
        if (canvas_size.x < 50) canvas_size.x = 50;
        if (canvas_size.y < 50) canvas_size.y = 50;
        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y), IM_COL32(30,30,40,255));
        draw->AddRect(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y), IM_COL32(100,100,100,255));

        if (!layer_sizes.empty()) {
            float spacing = canvas_size.x / (layer_sizes.size() + 1);
            const int max_display = 48;

            for (size_t i = 0; i < layer_sizes.size(); ++i) {
                float x = canvas_pos.x + (i + 1) * spacing;
                int n = layer_sizes[i];
                const char* type = i == 0 ? "Input" : (i == layer_sizes.size()-1 ? "Output" : "Hidden");

                if (n > max_display) {
                    float h = canvas_size.y * 0.8f;
                    float y = canvas_pos.y + (canvas_size.y - h) * 0.5f;
                    draw->AddRectFilled(ImVec2(x-15, y), ImVec2(x+15, y+h), IM_COL32(70,130,180,255));
                    char buf[32]; sprintf(buf, "%d", n);
                    draw->AddText(ImVec2(x-15, y + h*0.5f - 10), IM_COL32(255,255,255,255), buf);
                } else {
                    float step = canvas_size.y / (n + 1);
                    for (int j = 0; j < n; ++j) {
                        float y = canvas_pos.y + (j + 1) * step;
                        draw->AddCircleFilled(ImVec2(x, y), 8, IM_COL32(100,200,255,255));
                    }
                }
                char label[64];
                sprintf(label, "%s: %d", type, n);
                draw->AddText(ImVec2(x - 50, canvas_pos.y + 10), IM_COL32(200,200,255,255), label);
                if (i < layer_acts.size()) {
                    const char* act = layer_acts[i] == ACT_SIGMOID ? "Sigmoid" :
                                     layer_acts[i] == ACT_TANH ? "Tanh" :
                                     layer_acts[i] == ACT_RELU ? "ReLU" : "Softmax";
                    sprintf(label, "→ %s", act);
                    draw->AddText(ImVec2(x - 50, canvas_pos.y + canvas_size.y - 30), IM_COL32(150,255,150,255), label);
                }
            }

            // Connections
            for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
                float x1 = canvas_pos.x + (i + 1) * spacing;
                float x2 = canvas_pos.x + (i + 2) * spacing;
                int n1 = layer_sizes[i], n2 = layer_sizes[i+1];
                if (n1 > max_display || n2 > max_display) {
                    draw->AddLine(ImVec2(x1, canvas_pos.y + canvas_size.y*0.5f),
                                  ImVec2(x2, canvas_pos.y + canvas_size.y*0.5f),
                                  IM_COL32(180,180,180,200), 3.0f);
                } else {
                    float step1 = canvas_size.y / (n1 + 1);
                    float step2 = canvas_size.y / (n2 + 1);
                    for (int a = 0; a < n1; ++a)
                        for (int b = 0; b < n2; ++b)
                            draw->AddLine(ImVec2(x1, canvas_pos.y + (a+1)*step1),
                                          ImVec2(x2, canvas_pos.y + (b+1)*step2),
                                          IM_COL32(180,180,180,80), 1.0f);
                }
            }
        }
        ImGui::End();

        ImGui::End(); // DockSpace

        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(0.07f, 0.08f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    if (net) network_free(net);
    // Cleanup OpenGL/ImGui/GLFW...
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shader);
    glDeleteFramebuffers(1, &cubeFBO);
    glDeleteTextures(1, &cubeTexture);
    glDeleteRenderbuffers(1, &depthRB);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
