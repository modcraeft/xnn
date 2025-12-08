// plugins/logic.cpp

#include "../plugin.h"
#include <cstdio>

static bool show_window = true;

enum GateType {
    GATE_AND = 0, GATE_OR, GATE_XOR, GATE_NAND, GATE_NOR, GATE_XNOR,
    GATE_NOT, GATE_BUF, GATE_COUNT
};

struct GateInfo {
    const char* name;
    const char* symbol;
    bool is_two_input;
};

static const GateInfo gate_infos[GATE_COUNT] = {
    {"AND",     "&&",       true},  {"OR",      "||",       true},
    {"XOR",     "^",        true},  {"NAND",    "!(A&&B)",  true},
    {"NOR",     "!(A||B)",  true},  {"XNOR",    "!(A^B)",   true},
    {"NOT",     "!",        false}, {"BUFFER",  "A",        false}
};

static int  current_gate = GATE_XOR;
static bool input_a = false;
static bool input_b = false;

static bool get_output(bool a, bool b, GateType gate)
{
    switch (gate) {
        case GATE_AND:  return a && b;
        case GATE_OR:   return a || b;
        case GATE_XOR:  return a ^ b;
        case GATE_NAND: return !(a && b);
        case GATE_NOR:  return !(a || b);
        case GATE_XNOR: return !(a ^ b);
        case GATE_NOT:  return !a;
        case GATE_BUF:  return a;
        default:        return false;
    }
}

void imgui_plugin_init(ImGuiContext* ctx)
{
    ImGui::SetCurrentContext(ctx);
    printf("[Logic] Initialized – Logic Gate Data\n");
}

void imgui_plugin_update()
{
    if (!show_window) return;

    ImGui::SetNextWindowSize(ImVec2(480, 620), ImGuiCond_FirstUseEver);
    ImGui::Begin("Logic Gate Data", &show_window, ImGuiWindowFlags_NoCollapse);

    ImGui::Text("Logic Gate Training Data");
    ImGui::Separator();

    // Top controls
    if (ImGui::BeginCombo("Gate", gate_infos[current_gate].name)) {
        for (int i = 0; i < GATE_COUNT; ++i) {
            bool selected = (current_gate == i);
            if (ImGui::Selectable(gate_infos[i].name, selected))
                current_gate = i;
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    const bool is_two_input = gate_infos[current_gate].is_two_input;

    ImGui::Text("Inputs:");
    if (is_two_input) {
        ImGui::Checkbox("A", &input_a);
        ImGui::SameLine(100);
        ImGui::Checkbox("B", &input_b);
    } else {
        ImGui::Checkbox("A (Input)", &input_a);
        input_b = false;
    }

    const bool output = get_output(input_a, input_b, (GateType)current_gate);

    ImGui::Separator();
    ImGui::Text("Output: ");
    ImGui::SameLine();
    ImGui::TextColored(output ? ImVec4(0,1,0,1) : ImVec4(1,0.3f,0.3f,1),
                       "%s", output ? "TRUE (1)" : "FALSE (0)");
    ImGui::Text("Symbol: %s", gate_infos[current_gate].symbol);

    // Flexible canvas area – reserves space at the bottom for truth table
    const float bottom_height = 140.0f;
    ImGui::BeginChild("CanvasRegion", ImVec2(0, -bottom_height), false, 0);  // <-- fixed flags

    ImDrawList* draw = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos  = ImGui::GetCursorScreenPos();
    canvas_pos.x += 15; canvas_pos.y += 10;

    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    canvas_size.x -= 30;
    canvas_size.y -= 20;
    if (canvas_size.y < 180) canvas_size.y = 180;

    ImVec2 p0 = canvas_pos;

    const ImU32 col_on    = IM_COL32(0, 255, 0, 255);
    const ImU32 col_off   = IM_COL32(80, 80, 80, 255);
    const ImU32 col_gate  = IM_COL32(200, 200, 200, 255);
    const ImU32 col_curve = IM_COL32(255, 200, 0, 255);

    // Input wires
    if (is_two_input) {
        draw->AddLine(ImVec2(p0.x + 30, p0.y + canvas_size.y*0.25f),
                      ImVec2(p0.x + canvas_size.x*0.35f, p0.y + canvas_size.y*0.25f),
                      input_a ? col_on : col_off, 4.0f);
        draw->AddLine(ImVec2(p0.x + 30, p0.y + canvas_size.y*0.75f),
                      ImVec2(p0.x + canvas_size.x*0.35f, p0.y + canvas_size.y*0.75f),
                      input_b ? col_on : col_off, 4.0f);
    } else {
        draw->AddLine(ImVec2(p0.x + 30, p0.y + canvas_size.y*0.5f),
                      ImVec2(p0.x + canvas_size.x*0.35f, p0.y + canvas_size.y*0.5f),
                      input_a ? col_on : col_off, 4.0f);
    }

    // Output wire
    draw->AddLine(ImVec2(p0.x + canvas_size.x*0.70f, p0.y + canvas_size.y*0.5f),
                  ImVec2(p0.x + canvas_size.x - 30, p0.y + canvas_size.y*0.5f),
                  output ? col_on : col_off, 4.0f);

    // Gate body
    if (current_gate == GATE_NOT || current_gate == GATE_BUF) {
        ImVec2 a(p0.x + canvas_size.x*0.35f, p0.y + canvas_size.y*0.3f);
        ImVec2 b(p0.x + canvas_size.x*0.35f, p0.y + canvas_size.y*0.7f);
        ImVec2 c(p0.x + canvas_size.x*0.65f, p0.y + canvas_size.y*0.5f);
        draw->AddTriangleFilled(a, b, c, col_gate);
        if (current_gate == GATE_NOT)
            draw->AddCircleFilled(ImVec2(c.x + 18, c.y), 12, col_gate);
    } else {
        ImVec2 pts[4] = {
            ImVec2(p0.x + canvas_size.x*0.35f, p0.y + canvas_size.y*0.15f),
            ImVec2(p0.x + canvas_size.x*0.35f, p0.y + canvas_size.y*0.85f),
            ImVec2(p0.x + canvas_size.x*0.70f, p0.y + canvas_size.y*0.75f),
            ImVec2(p0.x + canvas_size.x*0.70f, p0.y + canvas_size.y*0.25f)
        };
        draw->AddConvexPolyFilled(pts, 4, col_gate);

        if (current_gate == GATE_OR || current_gate == GATE_NOR) {
            draw->AddBezierCubic(
                ImVec2(p0.x + canvas_size.x*0.35f, p0.y + canvas_size.y*0.15f),
                ImVec2(p0.x + canvas_size.x*0.50f, p0.y + canvas_size.y*0.05f),
                ImVec2(p0.x + canvas_size.x*0.70f, p0.y + canvas_size.y*0.50f),
                ImVec2(p0.x + canvas_size.x*0.50f, p0.y + canvas_size.y*0.95f),
                col_curve, 6.0f);
        }
        if (current_gate == GATE_XOR || current_gate == GATE_XNOR) {
            draw->AddBezierCubic(
                ImVec2(p0.x + canvas_size.x*0.30f, p0.y + canvas_size.y*0.15f),
                ImVec2(p0.x + canvas_size.x*0.45f, p0.y + canvas_size.y*0.05f),
                ImVec2(p0.x + canvas_size.x*0.65f, p0.y + canvas_size.y*0.50f),
                ImVec2(p0.x + canvas_size.x*0.45f, p0.y + canvas_size.y*0.95f),
                col_curve, 5.0f);
        }
        if (current_gate == GATE_NAND || current_gate == GATE_NOR || current_gate == GATE_XNOR) {
            draw->AddCircleFilled(ImVec2(p0.x + canvas_size.x*0.73f, p0.y + canvas_size.y*0.5f), 12, col_gate);
        }
    }

    // Labels
    ImGui::SetCursorScreenPos(ImVec2(p0.x + 8, p0.y + canvas_size.y*0.25f - 10));
    ImGui::TextColored(input_a ? ImVec4(0,1,0,1) : ImVec4(0.5f,0.5f,0.5f,1), "A");
    if (is_two_input) {
        ImGui::SetCursorScreenPos(ImVec2(p0.x + 8, p0.y + canvas_size.y*0.75f - 10));
        ImGui::TextColored(input_b ? ImVec4(0,1,0,1) : ImVec4(0.5f,0.5f,0.5f,1), "B");
    }
    ImGui::SetCursorScreenPos(ImVec2(p0.x + canvas_size.x - 28, p0.y + canvas_size.y*0.5f - 10));
    ImGui::TextColored(output ? ImVec4(0,1,0,1) : ImVec4(0.5f,0.5f,0.5f,1), "Y");

    ImGui::EndChild();

    // Bottom-docked Truth Table
    ImGui::Separator();
    ImGui::Text("Truth Table:");
    const int cols = is_two_input ? 3 : 2;
    if (ImGui::BeginTable("tt_bottom", cols,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
        ImGui::TableSetupColumn("A", ImGuiTableColumnFlags_WidthFixed, 60);
        if (is_two_input) ImGui::TableSetupColumn("B", ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableSetupColumn("Y", ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableHeadersRow();

        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < (is_two_input ? 2 : 1); ++b) {
                bool A = a, B = b;
                bool Y = get_output(A, B, (GateType)current_gate);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%d", A);
                if (is_two_input) { ImGui::TableSetColumnIndex(1); ImGui::Text("%d", B); }
                ImGui::TableSetColumnIndex(cols-1);
                ImGui::TextColored(Y ? ImVec4(0,1,0,1) : ImVec4(1,0.3f,0.3f,1), "%d", Y);
            }
        }
        ImGui::EndTable();
    }

    ImGui::End();
}

void imgui_plugin_shutdown()
{
    printf("[Logic] Shutdown – Logic Gate Data\n");
}
