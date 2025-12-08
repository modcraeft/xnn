// plugins/neural_net_viz.cpp
#include "../plugin.h"
#include <vector>
#include <string>
#include <cstdio>

static struct {
    bool show = true;
    std::vector<int> layer_sizes = {4, 8, 10, 6, 3};
    
    // Per-layer activation (index 0 = input, last = output)
    std::vector<std::string> activations = {
        "",        // Input: no activation
        "ReLU",    // Hidden
        "ReLU",
        "ReLU",
        "Softmax"  // Output (auto for classification)
    };
    
    // Global loss function
    enum class Loss { MSE, CrossEntropy };
    Loss loss = Loss::CrossEntropy;

    float connection_alpha = 0.45f;
    bool show_labels = true;
    bool compact_mode = false;
    const int max_neurons_display = 48;
} nn;

void imgui_plugin_init(ImGuiContext* ctx)
{
    ImGui::SetCurrentContext(ctx);
    printf("[Net [1]]\n");
}

void imgui_plugin_update()
{
    if (!nn.show) return;

    ImGui::SetNextWindowSize(ImVec2(1080, 720), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("[Net [1]]", &nn.show)) {
        ImGui::End();
        return;
    }

    // === Top Controls ===
    if (ImGui::Button("Add Hidden Layer")) {
        nn.layer_sizes.insert(nn.layer_sizes.end() - 1, 8);
        nn.activations.insert(nn.activations.end() - 1, "ReLU");
    }
    ImGui::SameLine();
    if (ImGui::Button("Remove Last") && nn.layer_sizes.size() > 2) {
        nn.layer_sizes.pop_back();
        nn.activations.pop_back();
    }
    ImGui::SameLine();
    if (ImGui::Button("Random Topology")) {
        for (size_t i = 1; i < nn.layer_sizes.size() - 1; ++i)
            nn.layer_sizes[i] = 4 + (rand() % 16);
    }

    ImGui::SameLine();
    ImGui::Checkbox("Compact", &nn.compact_mode);
    ImGui::SameLine();
    ImGui::Checkbox("Labels", &nn.show_labels);

    ImGui::SliderFloat("Conn Alpha", &nn.connection_alpha, 0.1f, 1.0f, "%.2f");

    // Loss Function Selector
    ImGui::Separator();
    ImGui::Text("Loss Function:");
    ImGui::SameLine();
    if (ImGui::RadioButton("Cross Entropy", nn.loss == nn.Loss::CrossEntropy)) {
        nn.loss = nn.Loss::CrossEntropy;
        nn.activations.back() = "Softmax";  // force softmax for classification
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("MSE", nn.loss == nn.Loss::MSE)) {
        nn.loss = nn.Loss::MSE;
        nn.activations.back() = "Linear";   // allow linear output for regression
    }

    // Per-Layer Controls
    ImGui::Separator();
    for (size_t i = 0; i < nn.layer_sizes.size(); ++i) {
        ImGui::PushID(i);
        int n = nn.layer_sizes[i];
        if (ImGui::DragInt("Neurons", &n, 0.3f, 1, 200)) {
            nn.layer_sizes[i] = std::max(1, n);
        }
        ImGui::SameLine();

        const char* type = i == 0 ? "Input" :
                          i == nn.layer_sizes.size()-1 ? "Output" : "Hidden";

        if (i == 0 || i == nn.layer_sizes.size()-1) {
            ImGui::Text("%s Layer: %d", type, nn.layer_sizes[i]);
        } else {
            ImGui::Text("Hidden Layer %zu: %d", i, nn.layer_sizes[i]);
        }

        if (i > 0) {
            ImGui::SameLine();
            const char* act = nn.activations[i].c_str();

            if (i == nn.layer_sizes.size()-1) {
                // Output layer: only Softmax (CE) or Linear (MSE)
                const char* out_act = (nn.loss == nn.Loss::CrossEntropy) ? "Softmax" : "Linear";
                ImGui::Text("→ %s", out_act);
            } else {
                // Hidden layer: full choice
                const char* items[] = {"ReLU", "Tanh", "Sigmoid", "Linear"};
                int current = 0;
                if (nn.activations[i] == "Tanh") current = 1;
                else if (nn.activations[i] == "Sigmoid") current = 2;
                else if (nn.activations[i] == "Linear") current = 3;

                if (ImGui::Combo("Act", &current, items, IM_ARRAYSIZE(items))) {
                    nn.activations[i] = items[current];
                }
            }
        }
        ImGui::PopID();
    }

    ImGui::Separator();

    // Canvas Rendering
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    if (canvas_size.x < 100) canvas_size.x = 100;
    if (canvas_size.y < 100) canvas_size.y = 100;

    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(canvas_pos,
                        ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                        IM_COL32(20, 22, 28, 255));
    draw->AddRect(canvas_pos,
                  ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                  IM_COL32(80, 80, 100, 255));

    if (nn.layer_sizes.empty()) { ImGui::End(); return; }

    float spacing_x = canvas_size.x / (nn.layer_sizes.size() + 1.0f);
    const float margin_y = 60.0f;

    // Draw layers + connections
    for (size_t i = 0; i < nn.layer_sizes.size(); ++i) {
        float x = canvas_pos.x + (i + 1) * spacing_x;
        int n = nn.layer_sizes[i];

        if (n > nn.max_neurons_display || nn.compact_mode) {
            float h = canvas_size.y - margin_y * 2;
            float y = canvas_pos.y + margin_y;
            ImU32 col = (i == 0) ? IM_COL32(80, 180, 80, 255) :
                       (i == nn.layer_sizes.size()-1) ? IM_COL32(180, 80, 180, 255) :
                       IM_COL32(70, 130, 220, 255);
            draw->AddRectFilled(ImVec2(x - 22, y), ImVec2(x + 22, y + h), col);
            draw->AddRect(ImVec2(x - 22, y), ImVec2(x + 22, y + h), IM_COL32(255,255,255,80), 0, 0, 2.5f);

            char buf[32]; snprintf(buf, sizeof(buf), "%d", n);
            ImVec2 ts = ImGui::CalcTextSize(buf);
            draw->AddText(ImVec2(x - ts.x * 0.5f, y + h * 0.5f - 10), IM_COL32(255,255,255,255), buf);
        } else {
            float step = (canvas_size.y - margin_y * 2) / (n + 1.0f);
            for (int j = 0; j < n; ++j) {
                float y = canvas_pos.y + margin_y + (j + 1) * step;
                ImU32 col = (i == 0) ? IM_COL32(100, 220, 100, 255) :
                           (i == nn.layer_sizes.size()-1) ? IM_COL32(220, 100, 220, 255) :
                           IM_COL32(100, 170, 255, 255);
                draw->AddCircleFilled(ImVec2(x, y), 13, col);
                draw->AddCircle(ImVec2(x, y), 13, IM_COL32(255,255,255,120), 32, 2.5f);
            }
        }

        if (nn.show_labels) {
            char label[64];
            const char* type = i == 0 ? "Input" : (i == nn.layer_sizes.size()-1 ? "Output" : "Hidden");
            snprintf(label, sizeof(label), "%s: %d", type, n);
            draw->AddText(ImVec2(x - 70, canvas_pos.y + 12), IM_COL32(200,220,255,255), label);

            if (i > 0) {
                const char* act = i == nn.layer_sizes.size()-1 ?
                    (nn.loss == nn.Loss::CrossEntropy ? "Softmax" : "Linear") :
                    nn.activations[i].c_str();
                snprintf(label, sizeof(label), "→ %s", act);
                draw->AddText(ImVec2(x - 70, canvas_pos.y + canvas_size.y - 38),
                              IM_COL32(150,255,150,255), label);
            }
        }
    }

    // Connections
    for (size_t i = 0; i < nn.layer_sizes.size() - 1; ++i) {
        float x1 = canvas_pos.x + (i + 1) * spacing_x;
        float x2 = canvas_pos.x + (i + 2) * spacing_x;
        int n1 = nn.layer_sizes[i], n2 = nn.layer_sizes[i+1];

        if (n1 > nn.max_neurons_display || n2 > nn.max_neurons_display || nn.compact_mode) {
            float y = canvas_pos.y + canvas_size.y * 0.5f;
            draw->AddLine(ImVec2(x1, y), ImVec2(x2, y),
                          IM_COL32(180,180,220,(int)(255*nn.connection_alpha)), 5.0f);
        } else {
            float step1 = (canvas_size.y - margin_y*2) / (n1 + 1.0f);
            float step2 = (canvas_size.y - margin_y*2) / (n2 + 1.0f);
            for (int a = 0; a < n1; ++a)
                for (int b = 0; b < n2; ++b) {
                    float y1 = canvas_pos.y + margin_y + (a + 1) * step1;
                    float y2 = canvas_pos.y + margin_y + (b + 1) * step2;
                    draw->AddLine(ImVec2(x1, y1), ImVec2(x2, y2),
                                  IM_COL32(180,180,200,(int)(90*nn.connection_alpha)), 1.0f);
                }
        }
    }

    ImGui::End();
}

void imgui_plugin_shutdown()
{
    printf("[neural_net_viz] Unloaded\n");
}
