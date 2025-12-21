// plugin.h
#pragma once
#include <imgui.h>
#include "xnn.h"  // Include xnn.h for Data and other types

#ifdef __cplusplus
extern "C" {
#endif

// Every plugin MUST export these three functions
void imgui_plugin_init(ImGuiContext* ctx);      // called once after dlopen
void imgui_plugin_update();                     // called every frame
void imgui_plugin_shutdown();                   // called before dlclose

extern "C" void register_data_provider(const char* name, Data* (*func)());
extern "C" Data* get_training_data(const char* name);

#ifdef __cplusplus
}
#endif
