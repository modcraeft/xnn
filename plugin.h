// plugin.h
#pragma once
#include <imgui.h>

#ifdef __cplusplus
extern "C" {
#endif

// Every plugin MUST export these three functions
void imgui_plugin_init(ImGuiContext* ctx);      // called once after dlopen
void imgui_plugin_update();                     // called every frame
void imgui_plugin_shutdown();                   // called before dlclose

#ifdef __cplusplus
}
#endif
