#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <dlfcn.h>
#include <filesystem>
#include <vector>
#include <string>
#include <stdio.h>
#include <unistd.h>


// Plugin system
struct Plugin {
    void* handle = nullptr;
    void (*init)(ImGuiContext*) = nullptr;
    void (*update)() = nullptr;
    void (*shutdown)() = nullptr;
    std::string path;
    time_t last_modified = 0;
};

std::vector<Plugin> plugins;

void load_plugin(const std::string& path)
{
    void* handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return;
    }

    auto get = [&](const char* name) { return dlsym(handle, name); };

    Plugin p;
    p.handle = handle;
    p.path = path;
    p.last_modified = std::filesystem::last_write_time(path).time_since_epoch().count();
    p.init     = (void(*)(ImGuiContext*))get("imgui_plugin_init");
    p.update   = (void(*)())get("imgui_plugin_update");
    p.shutdown = (void(*)())get("imgui_plugin_shutdown");

    if (p.init && p.update && p.shutdown) {
        p.init(ImGui::GetCurrentContext());
        plugins.push_back(std::move(p));
        printf("Loaded: %s\n", path.c_str());
    } else {
        dlclose(handle);
        fprintf(stderr, "Invalid plugin: %s (missing symbols)\n", path.c_str());
    }
}

void reload_changed_plugins()
{
    for (auto& p : plugins) {
        if (!p.handle) continue;
        auto new_time = std::filesystem::last_write_time(p.path).time_since_epoch().count();
        if (new_time != p.last_modified) {
            printf("Hot-reloading: %s\n", p.path.c_str());
            if (p.shutdown) p.shutdown();
            dlclose(p.handle);
            usleep(100000);  // tiny delay to let filesystem settle
            load_plugin(p.path);
        }
    }
}

void unload_all_plugins()
{
    for (auto& p : plugins) {
        if (p.shutdown) p.shutdown();
        if (p.handle) dlclose(p.handle);
    }
    plugins.clear();
}


// Theme
static const char* FONT_PATH = "fonts/hack.ttf";
static const float FONT_SIZE = 18.0f;

static void LoadCustomFont()
{
    ImGuiIO& io = ImGui::GetIO();
    ImFont* f = io.Fonts->AddFontFromFileTTF(FONT_PATH, FONT_SIZE);
    if (!f) { fprintf(stderr, "Font failed: default\n"); io.Fonts->AddFontDefault(); }
    io.Fonts->AddFontFromFileTTF(FONT_PATH, FONT_SIZE * 1.8f);
}

static void Theme_ProDark()
{
    ImGuiStyle& s = ImGui::GetStyle();
    ImVec4* c = s.Colors;
    s.WindowRounding = s.FrameRounding = s.GrabRounding = 2.0f;
    c[ImGuiCol_WindowBg]       = ImVec4(0.05f, 0.05f, 0.05f, 1.00f);
    c[ImGuiCol_TitleBgActive]  = ImVec4(0.15f, 0.35f, 0.65f, 1.00f);
    c[ImGuiCol_Button]         = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
    c[ImGuiCol_ButtonHovered]  = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    c[ImGuiCol_CheckMark]      = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    c[ImGuiCol_Text]           = ImVec4(0.95f, 0.95f, 0.95f, 1.00f);
}

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1600, 1000, "xnn", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    Theme_ProDark();
    LoadCustomFont();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Auto-load all .so in plugins/
    for (const auto& entry : std::filesystem::directory_iterator("plugins")) {
        if (entry.path().extension() == ".so") {
            load_plugin(entry.path().string());
        }
    }

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Built-in plugin manager
        ImGui::Begin("Plugin System", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Loaded plugins: %zu", plugins.size());
        if (ImGui::Button("Reload All Changed Plugins")) reload_changed_plugins();
        if (ImGui::Button("Unload All")) unload_all_plugins();
        ImGui::Text("Exit");
        ImGui::End();

        // Call all plugins
        for (auto& p : plugins)
            if (p.update) p.update();

        ImGui::Render();
        int w, h; glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    unload_all_plugins();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
