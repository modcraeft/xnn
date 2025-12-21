// plugins/net.cpp (updated with fixed y-axis and configurable history range)
#define XNN_IMPLEMENTATION  // Ensure xnn implementation is included
#include "../plugin.h"
#include <vector>
#include <string>
#include <cstdio>
#include <vector>  // For loss_history

// Assuming ImPlot is available; if not, use fallback ImGui::PlotLines
#include <implot.h>  // Ensure linked in Makefile

extern "C" Data* get_training_data(const char* name);

static struct {
    bool show = true;
    bool show_verify = true;  // Restored to true by default
    std::vector<size_t> layer_sizes = {2, 4, 1};  // Default for logic gates (e.g., XOR: 2 in, 1 out)
    std::vector<std::string> activations = {
        "",         // Input: no activation
        "Sigmoid",  // Hidden
        "Sigmoid"   // Output for binary
    };
    enum class Loss { MSE, CrossEntropy };
    Loss loss = Loss::CrossEntropy;
    float connection_alpha = 0.45f;
    bool show_labels = true;
    bool compact_mode = false;
    const int max_neurons_display = 48;

    // NN integration
    Network* network = nullptr;
    std::vector<int> act_ids;
    int loss_id = LOSS_CE;
    float learning_rate = 0.01f;
    int train_epochs = 1;  // Epochs per train action (or per frame in continuous mode)

    // Verification and continuous training
    std::vector<float> loss_history;  // History for graphing
    int max_history = 100;  // Configurable max points (changed from const to int for UI slider)
    float threshold = 0.5f;  // Configurable threshold for true/false
    bool is_training = false;  // New: Continuous training state
} nn;

static int get_act_id(const std::string& s) {
    if (s == "Sigmoid") return ACT_SIGMOID;
    if (s == "Tanh") return ACT_TANH;
    if (s == "ReLU") return ACT_RELU;
    if (s == "Softmax") return ACT_SOFTMAX;
    if (s == "Linear") return ACT_LINEAR;
    return ACT_RELU;  // Default
}

void imgui_plugin_init(ImGuiContext* ctx)
{
    ImGui::SetCurrentContext(ctx);
    ImPlot::CreateContext();  // Initialize ImPlot context
    printf("[Net] Initialized\n");
    XNN_INIT();
}

static void update_loss_history(float new_loss) {
    nn.loss_history.push_back(new_loss);
    if ((int)nn.loss_history.size() > nn.max_history) nn.loss_history.erase(nn.loss_history.begin());
}

static void perform_training(Data* training_data, Network* grad) {
    for (int ep = 0; ep < nn.train_epochs; ++ep) {
        backprop(nn.network, grad, training_data);
        apply_grad(nn.network, grad, nn.learning_rate);
        float current_loss = network_mse(nn.network, training_data);
        update_loss_history(current_loss);
    }
}

void imgui_plugin_update()
{
    // Main Network Window
    if (!nn.show) return;

    ImGui::SetNextWindowSize(ImVec2(1080, 720), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("[Net]", &nn.show)) {
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
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("MSE", nn.loss == nn.Loss::MSE)) {
        nn.loss = nn.Loss::MSE;
    }

    // Per-Layer Controls
    ImGui::Separator();
    for (size_t i = 0; i < nn.layer_sizes.size(); ++i) {
        ImGui::PushID(i);
        int temp = (int)nn.layer_sizes[i];
        // Set width to half the available space
        float avail_width = ImGui::GetContentRegionAvail().x;
        ImGui::SetNextItemWidth(avail_width * 0.5f);  // Half width for the drag bar
        if (ImGui::DragInt("Neurons", &temp, 0.3f, 1, 200)) {
            nn.layer_sizes[i] = (size_t)std::max(1, temp);
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
            int current = get_act_id(nn.activations[i]);
            const char* items[] = {"ReLU", "Tanh", "Sigmoid", "Softmax", "Linear"};
            int item_idx = 0;
            if (current == ACT_TANH) item_idx = 1;
            else if (current == ACT_SIGMOID) item_idx = 2;
            else if (current == ACT_SOFTMAX) item_idx = 3;
            else if (current == ACT_LINEAR) item_idx = 4;

            if (ImGui::Combo("Act", &item_idx, items, IM_ARRAYSIZE(items))) {
                nn.activations[i] = items[item_idx];
            }
        }
        ImGui::PopID();
    }

    ImGui::Separator();

    // NN Creation and Training
    nn.loss_id = (nn.loss == nn.Loss::CrossEntropy) ? LOSS_CE : LOSS_MSE;
    nn.act_ids.resize(nn.layer_sizes.size());
    nn.act_ids[0] = -1;  // Input
    for (size_t i = 1; i < nn.layer_sizes.size(); ++i) {
        nn.act_ids[i] = get_act_id(nn.activations[i]);
    }
    // Override output activation
    size_t out_size = nn.layer_sizes.back();
    if (nn.loss_id == LOSS_CE) {
        if (out_size == 1) {
            nn.act_ids.back() = ACT_SIGMOID;
            nn.activations.back() = "Sigmoid";
        } else {
            nn.act_ids.back() = ACT_SOFTMAX;
            nn.activations.back() = "Softmax";
        }
    } else {
        nn.act_ids.back() = ACT_LINEAR;
        nn.activations.back() = "Linear";
    }

    if (ImGui::Button("Create/Reset Network")) {
        if (nn.network) network_free(nn.network);
        nn.network = network_alloc(nn.layer_sizes.data(), nn.layer_sizes.size(), nn.act_ids.data(), nn.loss_id);
        nn.loss_history.clear();  // Reset history on reset
        nn.is_training = false;  // Stop continuous training on reset
    }

    ImGui::Separator();
    ImGui::Text("Training");
    ImGui::DragFloat("Learning Rate", &nn.learning_rate, 0.001f, 0.0001f, 1.0f);
    ImGui::DragInt("Epochs per Frame/Click", &nn.train_epochs, 1, 1, 100);  // Renamed for clarity

    Data* training_data = get_training_data("gate_training_data");
    if (!training_data) {
        ImGui::TextColored(ImVec4(1,0,0,1), "Training data unavailable. Load Logic plugin.");
        nn.is_training = false;  // Auto-pause if data missing
    } else if (!nn.network) {
        ImGui::TextColored(ImVec4(1,0.5f,0,1), "Create network first.");
        nn.is_training = false;
    } else if (training_data->in->cols != nn.layer_sizes[0] ||
               training_data->out->cols != nn.layer_sizes.back()) {
        ImGui::TextColored(ImVec4(1,0,0,1), "Size mismatch! Adjust network input/output.");
        nn.is_training = false;
    } else {
        ImGui::Text("Training data from Logic: %zu samples (%zu in, %zu out)",
                    training_data->in->rows, training_data->in->cols, training_data->out->cols);

        // Manual Train Button
        if (ImGui::Button("Train Once")) {
            Network* grad = network_alloc(nn.layer_sizes.data(), nn.layer_sizes.size(), nn.act_ids.data(), nn.loss_id);
            perform_training(training_data, grad);
            network_free(grad);
        }

        // Continuous Training Toggle
        ImGui::SameLine();
        if (ImGui::Button(nn.is_training ? "Pause Training" : "Start Continuous Training")) {
            nn.is_training = !nn.is_training;
        }

        // Continuous Training Logic (runs if active)
        if (nn.is_training) {
            Network* grad = network_alloc(nn.layer_sizes.data(), nn.layer_sizes.size(), nn.act_ids.data(), nn.loss_id);
            perform_training(training_data, grad);
            network_free(grad);
        }

        // Real-time loss computation (every frame)
        float current_loss = network_mse(nn.network, training_data);
        ImGui::Text("Current MSE Loss: %.6f", current_loss);
    }

    ImGui::Checkbox("Show Verification Window", &nn.show_verify);

    ImGui::Separator();

    // Canvas Rendering (restored and default)
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    if (canvas_size.x < 100) canvas_size.x = 100;
    if (canvas_size.y < 100) canvas_size.y = 100;

    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(canvas_pos,
                        ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                        IM_COL32(0, 0, 0, 255));
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
                draw->AddCircle(ImVec2(x, y), 13, IM_COL32(0.0f,0.0f,0.0f,255), 32, 2.5f);
            }
        }

        if (nn.show_labels) {
            char label[64];
            const char* type = i == 0 ? "Input" : (i == nn.layer_sizes.size()-1 ? "Output" : "Hidden");
            snprintf(label, sizeof(label), "%s: %d", type, n);
            draw->AddText(ImVec2(x - 70, canvas_pos.y + 12), IM_COL32(200,220,255,255), label);

            if (i > 0) {
                const char* act = nn.activations[i].c_str();
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

    // Separate Verification Window
    if (!nn.show_verify) return;

    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Network Verification", &nn.show_verify)) {
        ImGui::End();
        return;
    }

    training_data = get_training_data("gate_training_data");  // Re-fetch in case unloaded/reloaded
    if (!training_data) {
        ImGui::TextColored(ImVec4(1,0,0,1), "Training data unavailable. Load Logic plugin.");
        ImGui::End();
        return;
    }
    if (!nn.network) {
        ImGui::TextColored(ImVec4(1,0.5f,0,1), "No network created. Use Create/Reset Network.");
        ImGui::End();
        return;
    }

    ImGui::Text("Real-time Predictions vs. Expected (from Logic Data)");
    ImGui::Separator();

    // Predictions Table
    size_t samples = training_data->in->rows;
    size_t in_cols = training_data->in->cols;
    size_t out_cols = training_data->out->cols;
    if (ImGui::BeginTable("predictions", in_cols + out_cols * 2 + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
        for (size_t col = 0; col < in_cols; ++col) {
            ImGui::TableSetupColumn(("In " + std::to_string(col)).c_str(), ImGuiTableColumnFlags_WidthFixed, 60);
        }
        for (size_t col = 0; col < out_cols; ++col) {
            ImGui::TableSetupColumn(("Exp " + std::to_string(col)).c_str(), ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn(("Pred " + std::to_string(col)).c_str(), ImGuiTableColumnFlags_WidthFixed, 80);
        }
        ImGui::TableSetupColumn("Match?", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableHeadersRow();

        float input_buf[32];  // Max input size assumption
        float output_buf[32]; // Max output size assumption
        for (size_t s = 0; s < samples; ++s) {
            ImGui::TableNextRow();
            // Inputs
            for (size_t col = 0; col < in_cols; ++col) {
                ImGui::TableSetColumnIndex(col);
                ImGui::Text("%.1f", training_data->in->data[s * in_cols + col]);
            }
            // Expected vs Predicted per output
            memcpy(input_buf, &training_data->in->data[s * in_cols], in_cols * sizeof(float));
            network_predict(nn.network, input_buf, output_buf);
            bool all_match = true;
            for (size_t col = 0; col < out_cols; ++col) {
                float expected = training_data->out->data[s * out_cols + col];
                float predicted = output_buf[col];
                bool match = (predicted > nn.threshold) == (expected > nn.threshold);
                all_match &= match;

                ImGui::TableSetColumnIndex(in_cols + col * 2);
                ImGui::TextColored(expected > nn.threshold ? ImVec4(0,1,0,1) : ImVec4(1,0.3f,0.3f,1), "%.3f (%s)", expected, expected > nn.threshold ? "True" : "False");

                ImGui::TableSetColumnIndex(in_cols + col * 2 + 1);
                ImGui::TextColored(predicted > nn.threshold ? ImVec4(0,1,0,1) : ImVec4(1,0.3f,0.3f,1), "%.3f (%s)", predicted, predicted > nn.threshold ? "True" : "False");
            }
            // Match
            ImGui::TableSetColumnIndex(in_cols + out_cols * 2);
            ImGui::TextColored(all_match ? ImVec4(0,1,0,1) : ImVec4(1,0,0,1), "%s", all_match ? "Yes" : "No");
        }
        ImGui::EndTable();
    }

    ImGui::Separator();
    ImGui::Text("Loss History Graph");
    ImGui::SliderFloat("True/False Threshold", &nn.threshold, 0.0f, 1.0f, "%.2f");
    ImGui::DragInt("Max History Points", &nn.max_history, 1, 10, 10000);  // New slider for changing range
    if (ImGui::Button("Clear Loss History")) nn.loss_history.clear();

    // Loss Graph (using ImPlot; fallback to ImGui::PlotLines if ImPlot not available)
    if (ImPlot::BeginPlot("Loss Over Time", ImVec2(-1, 650))) {
        ImPlot::SetupAxes("Epoch/Step", "MSE Loss", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0f, 1.0f, ImPlotCond_Always);  // Fixed y-axis 0 to 1
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0f, (double)nn.loss_history.size(), ImPlotCond_Always);  // x-axis from 0 to current history size
        ImPlot::PlotLine("Loss", nn.loss_history.data(), nn.loss_history.size());
        ImPlot::EndPlot();
    }
    // Fallback if ImPlot not linked: 
    // ImGui::PlotLines("Loss", nn.loss_history.data(), nn.loss_history.size(), 0, nullptr, 0.0f, 1.0f, ImVec2(0, 200));  // Fixed min/max, increased height

    ImGui::End();
}

void imgui_plugin_shutdown()
{
    if (nn.network) network_free(nn.network);
    ImPlot::DestroyContext();  // Clean up ImPlot context
    printf("[Net] Unloaded\n");
}
