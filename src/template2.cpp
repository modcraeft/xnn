// src/main.cpp – WITH 3D CUBE IN IMGUI VIEWPORT
#include <stdio.h>
#include <math.h>

// 1. GLAD
#define GLAD_GL_IMPLEMENTATION
#include "glad/glad.h"

// 2. GLFW
#include <GLFW/glfw3.h>

// 3. ImGui
#define IMGUI_ENABLE_DOCKING
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

// 4. cglm for math (you already have this)
#define CGLM_CLIPSPACE_ZERO_TO_ONE // optional, matches OpenGL default depth
#include "cglm/cglm.h"

// Simple shader loading helper
static GLuint compile_shader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        printf("Shader compilation failed:\n%s\n", log);
    }
    return shader;
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(1600, 1000, "xnn – 3D Cube Demo", nullptr, nullptr);
    if (!window) return -1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    // === ImGui Setup ===
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // === 3D Cube Setup ===
    // Vertex data (positions + colors)
    float vertices[] = {
        // positions          // colors
        -0.5f, -0.5f, -0.5f,  1.0f, 0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f, 0.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, 1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.5f, 0.5f, 0.5f,
    };

    unsigned int indices[] = {
        0, 1, 2,  2, 3, 0, // back
        4, 5, 6,  6, 7, 4, // front
        0, 1, 5,  5, 4, 0, // bottom
        3, 2, 6,  6, 7, 3, // top
        0, 3, 7,  7, 4, 0, // left
        1, 2, 6,  6, 5, 1  // right
    };

    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Shaders
    const char* vertexSrc = R"(
        #version 130
        in vec3 aPos;
        in vec3 aColor;
        out vec3 color;
        uniform mat4 mvp;
        void main() {
            gl_Position = mvp * vec4(aPos, 1.0);
            color = aColor;
        }
    )";

    const char* fragmentSrc = R"(
        #version 130
        in vec3 color;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(color, 1.0);
        }
    )";

    GLuint shaderProgram = glCreateProgram();
    GLuint vs = compile_shader(vertexSrc, GL_VERTEX_SHADER);
    GLuint fs = compile_shader(fragmentSrc, GL_FRAGMENT_SHADER);
    glAttachShader(shaderProgram, vs);
    glAttachShader(shaderProgram, fs);
    glLinkProgram(shaderProgram);

    glDeleteShader(vs);
    glDeleteShader(fs);

    // Framebuffer for rendering cube into ImGui texture
    GLuint cubeFBO, cubeTexture;
    glGenFramebuffers(1, &cubeFBO);
    glGenTextures(1, &cubeTexture);
    glBindTexture(GL_TEXTURE_2D, cubeTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1024, 1024, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindFramebuffer(GL_FRAMEBUFFER, cubeFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, cubeTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    bool show_cube_window = true;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // === Control Panel ===
        ImGui::Begin("xnn Control Panel");
        ImGui::Text("xnn is now running with 3D rendering!");
        ImGui::Text("FPS: %.1f", io.Framerate);
        if (ImGui::Button("Success!")) puts("It works!");
        ImGui::End();

        // === 3D Cube Window (Dockable!) ===
        ImGui::Begin("3D Cube Viewport", &show_cube_window, ImGuiWindowFlags_NoScrollbar);

        ImVec2 viewportSize = ImGui::GetContentRegionAvail();
        if (viewportSize.x > 0 && viewportSize.y > 0) {
            // Resize texture if needed
            GLuint currentTexWidth = (GLuint)viewportSize.x;
            GLuint currentTexHeight = (GLuint)viewportSize.y;
            static GLuint lastW = 0, lastH = 0;
            if (currentTexWidth != lastW || currentTexHeight != lastH) {
                glBindTexture(GL_TEXTURE_2D, cubeTexture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, currentTexWidth, currentTexHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
                lastW = currentTexWidth;
                lastH = currentTexHeight;
            }

            // Render cube to FBO
            glBindFramebuffer(GL_FRAMEBUFFER, cubeFBO);
            glViewport(0, 0, (GLsizei)viewportSize.x, (GLsizei)viewportSize.y);
            glEnable(GL_DEPTH_TEST);
            glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glUseProgram(shaderProgram);
            mat4 model = {0}, view = {0}, proj = {0}, mvp = {0};
            glm_mat4_identity(model);
            glm_mat4_identity(view);
            glm_mat4_identity(proj);

            float time = (float)glfwGetTime();
            glm_rotate(model, time * glm_rad(45.0f), (vec3){0.5f, 1.0f, 0.0f});
            glm_lookat((vec3){2,2,2}, (vec3){0,0,0}, (vec3){0,1,0}, view);
            glm_perspective(glm_rad(45.0f), viewportSize.x / viewportSize.y, 0.1f, 100.0f, proj);

            glm_mat4_mul(proj, view, mvp);
            glm_mat4_mul(mvp, model, mvp);

            GLint mvpLoc = glGetUniformLocation(shaderProgram, "mvp");
            glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, (float*)mvp);

            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        // Show the rendered texture
        ImGui::Image((void*)(intptr_t)cubeTexture, viewportSize, ImVec2(0,1), ImVec2(1,0));
        ImGui::End();

        // === Rendering ===
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(0.07f, 0.08f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow* backup = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup);
        }

        glfwSwapBuffers(window);
    }

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);
    glDeleteFramebuffers(1, &cubeFBO);
    glDeleteTextures(1, &cubeTexture);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
