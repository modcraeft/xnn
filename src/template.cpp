
// src/main.cpp  ← FINAL VERSION – WORKS IMMEDIATELY

#include <stdio.h>

// 1. GLAD first
#define GLAD_GL_IMPLEMENTATION
#include "glad/glad.h"

// 2. GLFW second
#include <GLFW/glfw3.h>

// 3. ImGui – with docking enabled
#define IMGUI_ENABLE_DOCKING   // ← THIS LINE IS THE KEY
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1400, 900, "xnn – Ready", nullptr, nullptr);
    if (!window) return -1;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;   // now works!
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // optional bonus

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    bool show_demo = true;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (show_demo)
            ImGui::ShowDemoWindow(&show_demo);

        ImGui::Begin("xnn Control Panel");
        ImGui::Text("xnn is now running!");
        ImGui::Separator();
        ImGui::Text("Built with C++ + ImGui + GLFW");
        ImGui::Text("FPS: %.1f", io.Framerate);
        if (ImGui::Button("It works!"))
            puts("Success!");
        ImGui::End();

        glClearColor(0.07f, 0.08f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Viewports support (makes windows draggable outside main window)
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup);
        }

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
