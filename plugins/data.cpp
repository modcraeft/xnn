// plugins/example_tool.cpp
#include "../plugin.h"
#include <cstdio>
#include <cmath>
#include <ctime>

static bool show_window = true;
static float value = 42.0f;
static float vec[3] = {1.0f, 0.5f, 0.0f};

static float sine_getter(void* /*data*/, int idx)
{
    return sinf(idx * 0.1f + (float)ImGui::GetTime());
}

void imgui_plugin_init(ImGuiContext* ctx)
{
    ImGui::SetCurrentContext(ctx);
    printf("[Input] Initialized – context set\n");
}

void imgui_plugin_update()
{
    if (!show_window) return;

    ImGui::Begin("Data", &show_window,
                 ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Text("Specify Input");
    ImGui::Text("Select: Image Data, Sound Data, Text Data");

    ImGui::SliderFloat("Magic Value", &value, 0.0f, 100.0f);
    ImGui::ColorEdit3("Color", vec);

    // ← FIXED: proper function pointer + include <cmath>
    ImGui::PlotLines("Live Sine", sine_getter, nullptr, 100, 0, nullptr,
                     -1.0f, 1.0f, ImVec2(300, 80));

    if (ImGui::Button("Allocate memory")) {
        printf("LOCK! value = %.2f\n", value);
    }

    ImGui::End();
}

void imgui_plugin_shutdown()
{
    printf("[Input] Shutdown!\n");
}
