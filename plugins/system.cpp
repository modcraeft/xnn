// plugins/system.cpp
#define SYSTEM_DEBUG

#include "../plugin.h"
#include <imgui.h>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <dlfcn.h>

namespace fs = std::filesystem;

#ifdef SYSTEM_DEBUG
    #define SYS_LOG(...) printf(__VA_ARGS__)
#else
    #define SYS_LOG(...) (void)0
#endif

static bool show_window = true;

struct PluginInfo {
    void* handle = nullptr;
    void (*init)(ImGuiContext*) = nullptr;
    void (*update_func)() = nullptr;
    void (*shutdown)() = nullptr;
    std::string path;
    time_t last_modified = 0;
    bool loaded = false;
};

static std::vector<PluginInfo> available_plugins;
static ImGuiContext* g_ctx = nullptr;

static void ApplyProDarkTheme()
{
    auto& s = ImGui::GetStyle();
    auto* c = s.Colors;
    s.WindowRounding = s.FrameRounding = s.GrabRounding = 2.0f;

    c[ImGuiCol_WindowBg]              = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    c[ImGuiCol_TitleBgActive]         = ImVec4(0.15f, 0.35f, 0.65f, 1.00f);
    c[ImGuiCol_Button]                = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
    c[ImGuiCol_ButtonHovered]         = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    c[ImGuiCol_ButtonActive]          = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
    c[ImGuiCol_CheckMark]             = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    c[ImGuiCol_Text]                  = ImVec4(0.95f, 0.95f, 0.95f, 1.00f);
    c[ImGuiCol_FrameBg]               = ImVec4(0.10f, 0.10f, 0.10f, 0.54f);
    c[ImGuiCol_FrameBgHovered]        = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
    c[ImGuiCol_FrameBgActive]         = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
    c[ImGuiCol_Header]                = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
    c[ImGuiCol_HeaderHovered]         = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
    c[ImGuiCol_HeaderActive]          = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
}

static bool IsPluginLoaded(const std::string& path)
{
    for (const auto& p : available_plugins)
        if (p.loaded && p.path == path)
            return true;
    return false;
}

static void LoadPlugin(const std::string& path)
{
    if (IsPluginLoaded(path)) {
        SYS_LOG("[System] Already loaded: %s\n", fs::path(path).filename().c_str());
        return;
    }

    void* h = dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (!h) {
        SYS_LOG("[System] dlopen failed: %s → %s\n", fs::path(path).filename().c_str(), dlerror());
        return;
    }

    auto get = [&](const char* sym) { return dlsym(h, sym); };

    PluginInfo pi{};
    pi.handle = h;
    pi.path = path;
    try {
        pi.last_modified = fs::last_write_time(path).time_since_epoch().count();
    } catch (...) { pi.last_modified = 0; }

    pi.init        = (void(*)(ImGuiContext*))get("imgui_plugin_init");
    pi.update_func = (void(*)())             get("imgui_plugin_update");
    pi.shutdown    = (void(*)())             get("imgui_plugin_shutdown");

    if (!pi.init || !pi.update_func || !pi.shutdown) {
        SYS_LOG("[System] Missing symbols in %s\n", fs::path(path).filename().c_str());
        dlclose(h);
        return;
    }

    pi.init(g_ctx);
    pi.loaded = true;
    available_plugins.push_back(std::move(pi));
    SYS_LOG("[System] Loaded: %s\n", fs::path(path).filename().c_str());
}

static void UnloadPlugin(size_t i)
{
    if (i >= available_plugins.size()) return;
    auto& p = available_plugins[i];
    if (p.shutdown) p.shutdown();
    if (p.handle) dlclose(p.handle);
    available_plugins.erase(available_plugins.begin() + i);
}

static void ReloadPlugin(size_t i)
{
    if (i >= available_plugins.size()) return;
    std::string path = available_plugins[i].path;
    UnloadPlugin(i);
    LoadPlugin(path);
}

void imgui_plugin_init(ImGuiContext* ctx)
{
    g_ctx = ctx;
    ImGui::SetCurrentContext(ctx);
    ApplyProDarkTheme();

    static bool scanned = false;
    if (!scanned) {
        scanned = true;
        try {
            for (const auto& e : fs::directory_iterator("plugins")) {
                if (e.path().extension() != ".so") continue;
                if (e.path().filename() == "system.so") continue;
                available_plugins.push_back(PluginInfo{
                    .path = e.path().string(),
                    .last_modified = e.last_write_time().time_since_epoch().count()
                });
            }
        } catch (...) {}
    }
}

void imgui_plugin_update()
{
    if (!show_window) return;

    ImGui::SetNextWindowSize(ImVec2(600, 780), ImGuiCond_FirstUseEver);
    ImGui::Begin("System & Plugin Manager", &show_window, ImGuiWindowFlags_NoCollapse);

    ImGui::Text("xnn — modular neural network visualizer");
    ImGui::Separator();

    // Theme selector
    const char* current_theme = "Pro Dark";
    if (ImGui::BeginCombo("Theme", current_theme)) {
        if (ImGui::Selectable("Pro Dark", true))  ApplyProDarkTheme();
        if (ImGui::Selectable("Dark",      false)) ImGui::StyleColorsDark();
        if (ImGui::Selectable("Light",     false)) ImGui::StyleColorsLight();
        ImGui::EndCombo();
    }
    ImGui::SameLine(); ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Global UI theme");

    size_t loaded_count = std::count_if(available_plugins.begin(), available_plugins.end(),
                                        [](const auto& p){ return p.loaded; });
    ImGui::Text("Loaded: %zu  |  FPS: %.1f", loaded_count, ImGui::GetIO().Framerate);
    ImGui::Separator();

    // Available Plugins
    if (ImGui::CollapsingHeader("Available Plugins", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (size_t i = 0; i < available_plugins.size(); ++i) {
            if (available_plugins[i].loaded) continue;

            bool already_loaded = IsPluginLoaded(available_plugins[i].path);
            ImGui::PushID(i);
            ImGui::Text("%s", fs::path(available_plugins[i].path).filename().c_str());
            ImGui::SameLine(380);
            ImGui::BeginDisabled(already_loaded);
            if (ImGui::Button("Load")) LoadPlugin(available_plugins[i].path);
            ImGui::EndDisabled();
            if (already_loaded) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.5f,0.5f,0.5f,1.0f), "(loaded)");
            }
            ImGui::PopID();
        }
    }

    // Loaded Plugins
    if (ImGui::CollapsingHeader("Loaded Plugins", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (size_t i = 0; i < available_plugins.size(); ++i) {
            auto& p = available_plugins[i];
            if (!p.loaded) continue;

            bool modified = false;
            try {
                auto t = fs::last_write_time(p.path).time_since_epoch().count();
                modified = (t != p.last_modified);
            } catch (...) {}

            ImGui::PushID(i);
            ImGui::Text("%s", fs::path(p.path).filename().c_str());
            if (modified) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1,1,0,1), "(modified)");
            }
            ImGui::SameLine(380);
            if (ImGui::Button("Unload")) { UnloadPlugin(i); ImGui::PopID(); continue; }
            ImGui::SameLine();
            if (ImGui::Button("Reload")) ReloadPlugin(i);
            ImGui::PopID();
        }

        ImGui::Separator();
        if (ImGui::Button("Reload All Changed")) {
            for (size_t i = 0; i < available_plugins.size(); ++i) {
                if (!available_plugins[i].loaded) continue;
                try {
                    auto t = fs::last_write_time(available_plugins[i].path).time_since_epoch().count();
                    if (t != available_plugins[i].last_modified) ReloadPlugin(i);
                } catch (...) {}
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Unload All")) {
            while (!available_plugins.empty() && available_plugins.back().loaded)
                UnloadPlugin(available_plugins.size() - 1);
        }
    }

    ImGui::End();

    // CALL ALL LOADED PLUGINS
    for (const auto& p : available_plugins) {
        if (p.loaded && p.update_func) {
            p.update_func();
        }
    }
}

void imgui_plugin_shutdown()
{
    while (!available_plugins.empty() && available_plugins.back().loaded)
        UnloadPlugin(available_plugins.size() - 1);
}
