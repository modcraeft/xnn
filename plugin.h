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

// Data passing API
extern "C" void register_data_provider(const char* name, Data* (*func)(), const char* plugin_name);  // Updated signature with plugin_name
extern "C" Data* get_training_data(const char* name);

// New: Get list of available provider names (updated signature to avoid type issues)
extern "C" void get_provider_names(const char*** out_names, int* count);  // Fills out_names with array, sets count

#ifdef __cplusplus
}
#endif
