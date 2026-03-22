// ui.cpp  —  reworked
//  Changes from original:
//   • syncsettings = true wired up after every widget that touches a settings field.
//     syncSettings() is called once per frame at the end (batched, not per-widget).
//   • Duplicates removed: Speed / Substeps were in Quick + World; now Quick only.
//     Wall force / wall dst were in Quick + Fluid; now Fluid → Forces only.
//   • Each tab is a pure DrawXxxContent() function — no giant monolith.
//   • Hybrid detachable tabs: [^] button floats a tab into a free-floating window;
//     [pin] button re-docks it into the sidebar.  Both modes share the same
//     DrawXxxContent() body so behaviour is identical.
//   • SYNC macro: one line after every widget that modifies settings.

#include <iostream>
#include <cstdarg>
#include "ui.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <algorithm>
#include "settings.h"
#include "compute.h"
#include "main.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Global sync flag
//  Set to true by any UI widget that modifies a settings field.
//  ui_init() calls syncSettings() at end of frame if true, then clears it.
// ─────────────────────────────────────────────────────────────────────────────
bool syncsettings = false;

// SYNC — drop this after any ImGui widget that writes to settings.
// Uses IsItemEdited() so it fires on every drag/key frame, not just on release.
#define SYNC do { if (ImGui::IsItemEdited()) syncsettings = true; } while(0)

static inline float UIClamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float UIMax(float a, float b) { return a > b ? a : b; }

// ─────────────────────────────────────────────────────────────────────────────
//  Panel geometry
// ─────────────────────────────────────────────────────────────────────────────
static const float PANEL_W = 340.0f;

// ─────────────────────────────────────────────────────────────────────────────
//  Detachable-tab state
//  s_det[i]     — true when tab i is floating as an independent window
//  s_needPos[i] — true once after detach, so we set the initial float position
// ─────────────────────────────────────────────────────────────────────────────
enum {
    TAB_QUICK = 0, TAB_FLUID, TAB_PARTICLES,
    TAB_WORLD, TAB_RENDER, TAB_PERF, TAB_HELP,
    TAB_COUNT
};
static const char* kTab[TAB_COUNT] = {
    "Quick", "Fluid", "Particles", "World", "Render", "Perf", "Help"
};
static bool s_det[TAB_COUNT] = {};
static bool s_needPos[TAB_COUNT] = { true,true,true,true,true,true,true };

// ─────────────────────────────────────────────────────────────────────────────
//  Theme  —  mid-grey, boxy, calm
// ─────────────────────────────────────────────────────────────────────────────
static void ApplyTheme()
{
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = 0.0f;  s.ChildRounding = 0.0f;
    s.FrameRounding = 2.0f;  s.PopupRounding = 2.0f;
    s.ScrollbarRounding = 2.0f;  s.GrabRounding = 2.0f;
    s.TabRounding = 2.0f;

    s.WindowPadding = ImVec2(10, 8);   s.FramePadding = ImVec2(6, 3);
    s.ItemSpacing = ImVec2(6, 4);    s.ItemInnerSpacing = ImVec2(4, 3);
    s.IndentSpacing = 14.0f;          s.ScrollbarSize = 9.0f;
    s.GrabMinSize = 8.0f;           s.WindowBorderSize = 1.0f;
    s.FrameBorderSize = 1.0f;           s.TabBorderSize = 0.0f;
    s.SeparatorTextBorderSize = 1.0f;      s.SeparatorTextPadding = ImVec2(6, 2);

    ImVec4* c = s.Colors;
    c[ImGuiCol_Text] = ImVec4(0.87f, 0.87f, 0.85f, 1.00f);
    c[ImGuiCol_TextDisabled] = ImVec4(0.53f, 0.53f, 0.51f, 1.00f);
    c[ImGuiCol_WindowBg] = ImVec4(0.18f, 0.18f, 0.18f, 0.98f);
    c[ImGuiCol_ChildBg] = ImVec4(0.21f, 0.21f, 0.21f, 1.00f);
    c[ImGuiCol_PopupBg] = ImVec4(0.18f, 0.18f, 0.18f, 0.97f);
    c[ImGuiCol_Border] = ImVec4(0.36f, 0.36f, 0.35f, 1.00f);
    c[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    c[ImGuiCol_FrameBg] = ImVec4(0.26f, 0.26f, 0.26f, 1.00f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.32f, 0.32f, 0.32f, 1.00f);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
    c[ImGuiCol_TitleBg] = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    c[ImGuiCol_TitleBgCollapsed] = ImVec4(0.15f, 0.15f, 0.15f, 0.80f);
    c[ImGuiCol_MenuBarBg] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
    c[ImGuiCol_ScrollbarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    c[ImGuiCol_ScrollbarGrab] = ImVec4(0.40f, 0.40f, 0.39f, 1.00f);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.52f, 0.52f, 0.51f, 1.00f);
    c[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.64f, 0.64f, 0.63f, 1.00f);
    c[ImGuiCol_CheckMark] = ImVec4(0.66f, 0.82f, 0.94f, 1.00f);
    c[ImGuiCol_SliderGrab] = ImVec4(0.54f, 0.70f, 0.84f, 1.00f);
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.70f, 0.84f, 0.96f, 1.00f);
    c[ImGuiCol_Button] = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
    c[ImGuiCol_ButtonActive] = ImVec4(0.52f, 0.52f, 0.52f, 1.00f);
    c[ImGuiCol_Header] = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
    c[ImGuiCol_HeaderActive] = ImVec4(0.48f, 0.48f, 0.48f, 1.00f);
    c[ImGuiCol_Separator] = ImVec4(0.36f, 0.36f, 0.35f, 1.00f);
    c[ImGuiCol_SeparatorHovered] = ImVec4(0.54f, 0.54f, 0.53f, 1.00f);
    c[ImGuiCol_SeparatorActive] = ImVec4(0.70f, 0.70f, 0.69f, 1.00f);
    c[ImGuiCol_ResizeGrip] = ImVec4(0.34f, 0.34f, 0.34f, 1.00f);
    c[ImGuiCol_ResizeGripHovered] = ImVec4(0.52f, 0.52f, 0.51f, 1.00f);
    c[ImGuiCol_ResizeGripActive] = ImVec4(0.68f, 0.68f, 0.67f, 1.00f);
    c[ImGuiCol_Tab] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
    c[ImGuiCol_TabHovered] = ImVec4(0.34f, 0.34f, 0.34f, 1.00f);
    c[ImGuiCol_TabActive] = ImVec4(0.28f, 0.28f, 0.28f, 1.00f);
    c[ImGuiCol_TabUnfocused] = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    c[ImGuiCol_TabUnfocusedActive] = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
    c[ImGuiCol_PlotLines] = ImVec4(0.58f, 0.68f, 0.76f, 1.00f);
    c[ImGuiCol_PlotLinesHovered] = ImVec4(0.74f, 0.84f, 0.92f, 1.00f);
    c[ImGuiCol_PlotHistogram] = ImVec4(0.50f, 0.62f, 0.72f, 1.00f);
    c[ImGuiCol_PlotHistogramHovered] = ImVec4(0.66f, 0.76f, 0.86f, 1.00f);
    c[ImGuiCol_TableHeaderBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    c[ImGuiCol_TableBorderStrong] = ImVec4(0.36f, 0.36f, 0.35f, 1.00f);
    c[ImGuiCol_TableBorderLight] = ImVec4(0.28f, 0.28f, 0.27f, 1.00f);
    c[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    c[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.04f);
    c[ImGuiCol_TextSelectedBg] = ImVec4(0.38f, 0.50f, 0.60f, 0.50f);
    c[ImGuiCol_NavHighlight] = ImVec4(0.58f, 0.72f, 0.84f, 1.00f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Widget helpers
// ─────────────────────────────────────────────────────────────────────────────
static void Sec(const char* label)
{
    ImGui::Spacing();
    ImGui::SeparatorText(label);
}

static void Badge(const char* label, ImVec4 col)
{
    ImGui::PushStyleColor(ImGuiCol_Button, col);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, col);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, col);
    ImGui::SmallButton(label);
    ImGui::PopStyleColor(3);
}

static bool DangerButton(const char* label)
{
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.40f, 0.14f, 0.14f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.52f, 0.20f, 0.20f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.64f, 0.26f, 0.26f, 1));
    bool p = ImGui::Button(label, ImVec2(-1, 0));
    ImGui::PopStyleColor(3);
    return p;
}

static void StatRow(const char* label, const char* fmt, ...)
{
    char buf[128];
    va_list a; va_start(a, fmt); vsnprintf(buf, sizeof(buf), fmt, a); va_end(a);
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0); ImGui::TextDisabled("%s", label);
    ImGui::TableSetColumnIndex(1); ImGui::Text("%s", buf);
}

static bool BeginStat2(const char* id)
{
    return ImGui::BeginTable(id, 2,
        ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Detach / pin helpers
// ─────────────────────────────────────────────────────────────────────────────

// Right-aligned [^] button that detaches tab id into a floating window.
// Call this as the FIRST item inside BeginChild / BeginTabItem content.
static void DetachButton(int id)
{
    const char* lbl = "[^]";
    float bw = ImGui::CalcTextSize(lbl).x + ImGui::GetStyle().FramePadding.x * 2.f + 2.f;
    float rx = ImGui::GetContentRegionMax().x - bw;
    ImGui::SetCursorPosX(rx);

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.00f, 0.00f, 0.00f, 0.00f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.38f, 0.42f, 0.48f, 0.70f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.52f, 0.56f, 0.62f, 1.00f));

    char bid[16]; snprintf(bid, sizeof(bid), "[^]##D%d", id);
    if (ImGui::SmallButton(bid)) {
        s_det[id] = true;
        s_needPos[id] = true;
    }
    ImGui::PopStyleColor(3);
    ImGui::SetItemTooltip(
        "Float this panel as a movable window.\n"
        "Click [pin] in the floating window to dock it back.");

    ImGui::Separator();
}

// [pin] button + hint text — call at TOP of a floating window's content.
static void PinButton(int id)
{
    if (ImGui::SmallButton("[pin]")) s_det[id] = false;
    ImGui::SetItemTooltip("Dock back into the sidebar tab bar.");
    ImGui::SameLine();
    ImGui::TextDisabled("floating — drag title to move");
    ImGui::Separator();
}

// Set initial position of a newly detached floating window, staggered by id.
static void PrepareFloatWindow(int id)
{
    if (s_needPos[id]) {
        ImGuiIO& io = ImGui::GetIO();
        float ox = ImGui::GetIO().DisplaySize.x * 0.25f + (float)id * 24.f;
        float oy = 70.f + (float)id * 18.f;
        // Keep inside screen bounds
        ox = UIClamp(ox, 10.f, io.DisplaySize.x - 340.f);
        oy = UIClamp(oy, 10.f, io.DisplaySize.y - 520.f);
        ImGui::SetNextWindowPos(ImVec2(ox, oy), ImGuiCond_Always);
        s_needPos[id] = false;
    }
    ImGui::SetNextWindowSize(ImVec2(330.f, 500.f), ImGuiCond_Once);
    ImGui::SetNextWindowBgAlpha(0.97f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  FPS ring buffer
// ─────────────────────────────────────────────────────────────────────────────
static float s_ring[128] = {};
static int   s_head = 0;

// ─────────────────────────────────────────────────────────────────────────────
//  Particle buffer fill bar (reused in Quick + Particles + Perf)
// ─────────────────────────────────────────────────────────────────────────────
static void FillBar(float h = 5.f)
{
    float fill = settings.maxparticles > 0
        ? (float)settings.count / (float)settings.maxparticles : 0.f;
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
        fill > 0.88f ? ImVec4(0.58f, 0.24f, 0.24f, 1) : ImVec4(0.34f, 0.42f, 0.50f, 1));
    ImGui::ProgressBar(fill, ImVec2(-1, h), "");
    ImGui::PopStyleColor();
    ImGui::SetItemTooltip("Buffer fill  %d / %d  (%.0f%%)",
        settings.count, settings.maxparticles, fill * 100.f);
}

// ═════════════════════════════════════════════════════════════════════════════
//  TAB CONTENT FUNCTIONS
//  Pure content — no Begin/End, no scroll child, no tab wrappers.
//  Called identically whether tab is docked or floating.
// ═════════════════════════════════════════════════════════════════════════════

// ─── QUICK ───────────────────────────────────────────────────────────────────
static void DrawQuickContent()
{
    // ── Simulation control ───────────────────────────────────────────────────
    Sec("Simulation");
    {
        bool paused = !settings.nopause;
        const char* lbl = paused ? "Resume  (Space)" : "Pause  (Space)";
        ImVec4 bc = paused ? ImVec4(0.38f, 0.16f, 0.16f, 1) : ImVec4(0.20f, 0.20f, 0.20f, 1);
        ImVec4 bh = paused ? ImVec4(0.50f, 0.22f, 0.22f, 1) : ImVec4(0.28f, 0.28f, 0.28f, 1);
        ImGui::PushStyleColor(ImGuiCol_Button, bc);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bh);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.44f, 0.44f, 0.44f, 1));
        if (ImGui::Button(lbl, ImVec2(-1, 0))) {
            settings.nopause = !settings.nopause;
            syncsettings = true;
        }
        ImGui::PopStyleColor(3);
        ImGui::SetItemTooltip("Toggle pause / resume.  Same as Space.");
    }

    ImGui::Spacing();
    ImGui::SliderFloat("Speed##q", &settings.simspeed, 0.001f, 10.0f, "%.2f"); SYNC;
    ImGui::SetItemTooltip("dt multiplier.  1.0 = real time.  0.2 = slow-mo.  3.0 = fast-fwd.");
    ImGui::InputInt("Substeps##q", &settings.substeps); SYNC;
    if (settings.substeps < 1) settings.substeps = 1;
    ImGui::SetItemTooltip("Physics steps per frame.  Higher = more stable at large k, but costs linearly more GPU time.");

    ImGui::Checkbox("Simulate long run##q", &settings.recordSim); SYNC;
    ImGui::SetItemTooltip("Renders every frame without skipping.  Useful for recording or high-count simulations.");

    // ── Key physics (quick-access; full detail in Fluid tab) ─────────────────
    Sec("Key Physics");
    ImGui::TextDisabled("Full detail in Fluid tab.");
    ImGui::Spacing();

    ImGui::SliderFloat("Gravity##q", &settings.gravityforce, 0.0f, 1000.0f, "%.0f"); SYNC;
    ImGui::SetItemTooltip("Constant downward acceleration per step.  0 = weightless.  150 = default.");

    if (ImGui::SliderFloat("h##q", &settings.h, 0.1f, 20.0f, "%.2f")) {
        calcKernels();
        syncsettings = true;
    }
    ImGui::SetItemTooltip("Smoothing radius.  All kernel coefficients update automatically.");

    ImGui::DragFloat("Rest rho##q", &settings.rest_density, 0.5f, 0.0f, 10000.f, "%.1f"); SYNC;
    ImGui::SetItemTooltip("Target equilibrium density.  Higher = particles draw together.");

    ImGui::DragFloat("Stiffness k##q", &settings.pressure, 10.0f, 0.0f, 200000.f, "%.0f"); SYNC;
    ImGui::SetItemTooltip("Compression resistance.  Start low, increase gradually.");

    ImGui::DragFloat("Near k'##q", &settings.nearpressure, 1.0f, 0.0f, 200000.f, "%.0f"); SYNC;
    ImGui::SetItemTooltip("Short-range repulsion.  Typically 10-50x stiffness k.");

    ImGui::DragFloat("Viscosity##q", &settings.visc, 0.01f, 0.0f, 200.f, "%.3f"); SYNC;
    ImGui::SetItemTooltip("Velocity averaging between neighbours.  Low = water.  High = honey.");

    // ── Toggles ──────────────────────────────────────────────────────────────
    Sec("Toggles");
    ImGui::Checkbox("SPH forces##q", &settings.colisionFun); SYNC;
    ImGui::SetItemTooltip("Enable density + pressure force kernels.");
    ImGui::SameLine(140);
    ImGui::Checkbox("Heat colour##q", &settings.heateffect);  SYNC;
    ImGui::SetItemTooltip("Shift particle colour by kinetic energy.  Blue = slow.  Red = fast.");

    // ── Restart ──────────────────────────────────────────────────────────────
    ImGui::Spacing();
    if (DangerButton("Restart"))
        restartSimulation();
    ImGui::SetItemTooltip("Wipe all particles and respawn with current settings.");
}

// ─── FLUID ───────────────────────────────────────────────────────────────────
static void DrawFluidContent()
{
    // ── Kernel ───────────────────────────────────────────────────────────────
    Sec("Kernel");
    if (ImGui::SliderFloat("h##fl", &settings.h, 0.1f, 20.0f, "%.2f")) {
        calcKernels();
        syncsettings = true;
    }
    ImGui::SetItemTooltip(
        "Interaction radius.  All coefficients derived from this.\n"
        "Larger h = thicker, more cohesive fluid.");

    if (ImGui::TreeNodeEx("Coefficients  (read-only)", ImGuiTreeNodeFlags_SpanFullWidth))
    {
        ImGui::BeginDisabled();
        ImGui::InputFloat("poly6", &settings.pollycoef6, 0, 0, "%.4e");
        ImGui::InputFloat("spiky", &settings.spikycoef, 0, 0, "%.4e");
        ImGui::InputFloat("spiky grad", &settings.spikygradv, 0, 0, "%.4e");
        ImGui::InputFloat("visc lap", &settings.viscosity, 0, 0, "%.4e");
        ImGui::InputFloat("self rho", &settings.Sdensity, 0, 0, "%.4e");
        ImGui::InputFloat("near self", &settings.ndensity, 0, 0, "%.4e");
        ImGui::EndDisabled();
        ImGui::TreePop();
    }

    // ── Density ──────────────────────────────────────────────────────────────
    Sec("Density");
    ImGui::DragFloat("Rest rho##fl", &settings.rest_density, 0.01f, 0.f, 10000.f, "%.2f"); SYNC;
    ImGui::SetItemTooltip("Target equilibrium density.  Raise to attract particles.  Lower to spread them.");

    // ── Pressure ─────────────────────────────────────────────────────────────
    Sec("Pressure");
    ImGui::DragFloat("Stiffness k##fl", &settings.pressure, 1.f, 0.f, 200000.f, "%.0f"); SYNC;
    ImGui::SetItemTooltip("Compression resistance.  Too high = instability.  Start ~100, increase gradually.");
    ImGui::DragFloat("Near k'##fl", &settings.nearpressure, 1.f, 0.f, 200000.f, "%.0f"); SYNC;
    ImGui::SetItemTooltip("Short-range repulsion.  Prevents collapse at close range.  Typically 10-50x k.");

    
    // ── Viscosity ─────────────────────────────────────────────────────────────
    Sec("Viscosity");
    ImGui::DragFloat("Viscosity##fl", &settings.visc, 0.01f, 0.f, 200.f, "%.3f"); SYNC;
    ImGui::SetItemTooltip("Velocity averaging between neighbours.  Low = water.  High = honey / thick fluid.");

    // ── Forces ───────────────────────────────────────────────────────────────
    Sec("Forces");
    ImGui::SliderFloat("Gravity##fl", &settings.gravityforce, 0.f, 1000.f, "%.0f"); SYNC;
    ImGui::SetItemTooltip("Downward acceleration each step.");
    ImGui::InputFloat("Restitution##fl", &settings.restitution, 0.02f, 0.1f, "%.3f"); SYNC;
    ImGui::SetItemTooltip("Wall bounce coefficient.  0.0 = fully inelastic.  1.0 = perfectly elastic.");
    ImGui::DragFloat("Wall force##fl", &settings.wallrep, 0.1f, 0.f, 10000.f, "%.0f"); SYNC;
    ImGui::SetItemTooltip("Repulsive force magnitude applied near bounding box walls.");
    ImGui::DragFloat("Wall dist##fl", &settings.walldst, 0.01f, 0.f, 10.f, "%.2f"); SYNC;
    ImGui::SetItemTooltip("Distance from wall at which repulsion kicks in.");

    // ── Pipeline toggles ──────────────────────────────────────────────────────
    Sec("Pipeline");
    ImGui::Checkbox("SPH forces##fl", &settings.colisionFun); SYNC;
    ImGui::SetItemTooltip("Master toggle for density + pressure force kernels.\nDisable to watch purely gravity-driven motion.");
}

// ─── PARTICLES ───────────────────────────────────────────────────────────────
static void DrawParticlesContent()
{
    // ── Spawn ─────────────────────────────────────────────────────────────────
    Sec("Spawn");
    ImGui::InputInt("Count##pt", &settings.totalBodies);
    if (ImGui::IsItemDeactivatedAfterEdit()) { restartSimulation(); syncsettings = true; }
    ImGui::SetItemTooltip("Particles spawned on restart.  Press Enter to confirm — restarts automatically.");

    ImGui::InputInt("Buffer##pt", &settings.maxparticles);
    if (ImGui::IsItemDeactivatedAfterEdit()) { restartSimulation(); syncsettings = true; }
    ImGui::SetItemTooltip("GPU allocation size in particles.  Must be >= Count.  Increase before using the emitter.");
   

    FillBar(5.f);

    ImGui::Spacing();
    ImGui::InputFloat("Radius##pt", &settings.size, 0.05f, 0.2f, "%.3f");
    if (ImGui::IsItemDeactivatedAfterEdit()) { restartSimulation(); syncsettings = true; }
    ImGui::SetItemTooltip("Visual + collision radius.  Restart required.");

    ImGui::InputFloat("Mass##pt", &settings.particleMass, 0.1f, 0.5f, "%.3f");
    if (ImGui::IsItemDeactivatedAfterEdit()) { restartSimulation(); syncsettings = true; }
    ImGui::SetItemTooltip("Particle mass.  Affects pressure and density contributions.");

    // ── Spawn tint ────────────────────────────────────────────────────────────
    

    // ── Heat colour ───────────────────────────────────────────────────────────
    Sec("Heat Colour");
    ImGui::Checkbox("Enable##pth", &settings.heateffect); SYNC;
    ImGui::SetItemTooltip("Shift particle colour by kinetic energy.  Blue = slow.  Red = fast.\n(Invisible in screen-space water render mode — switch to Particles mode.)");
    if (settings.heateffect)
    {
        ImGui::SliderFloat("Gen##pth", &settings.heatMultiplier, 0.1f, 20.f, "%.1f"); SYNC;
        ImGui::SetItemTooltip("How quickly movement raises the heat colour.");
        ImGui::SliderFloat("Fade##pth", &settings.cold, 0.1f, 20.f, "%.1f"); SYNC;
        ImGui::SetItemTooltip("Decay rate — how fast colour cools when idle.");
    }

    // ── Emitter ───────────────────────────────────────────────────────────────
    Sec("Emitter");
    ImGui::Checkbox("Enable##ptem", &settings.addParticle); SYNC;
    ImGui::SetItemTooltip("Inject new particles each frame until buffer is full.");
    ImGui::InputFloat("Max frame time##ptem", &settings.maxframetime, 0.f, 0.f, "%.1f"); SYNC;
    ImGui::SetItemTooltip("Cap injection when frame time exceeds this (ms).  Prevents emitter from stalling at high particle counts.");
    if (settings.addParticle)
    {
        ImGui::InputInt("Flow##ptem", &settings.flowcount); SYNC;
        ImGui::SetItemTooltip("Particles injected per frame.  Keep low (1-20) to avoid sudden buffer overflow.");
        ImGui::TextDisabled("emitted  %d", settings.samplecount);
    }

    
}

// ─── WORLD ───────────────────────────────────────────────────────────────────
static void DrawWorldContent()
{
    // ── Bounding box ──────────────────────────────────────────────────────────
    Sec("Bounding Box");
    ImGui::TextDisabled("Drag — updates in real time.");
    ImGui::Spacing();

    bool bchg = false;
    if (ImGui::BeginTable("##wdbb", 2, ImGuiTableFlags_SizingStretchSame))
    {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        bchg |= ImGui::DragFloat("+X##wd", &settings.maxX, 1.f, 0.f, 2000.f, "%.0f");
        ImGui::SetItemTooltip("Right wall X.  Drag left to shrink.");
        ImGui::TableSetColumnIndex(1);
        bchg |= ImGui::DragFloat("-X##wd", &settings.minX, 1.f, -2000.f, 0.f, "%.0f");
        ImGui::SetItemTooltip("Left wall X.");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        bchg |= ImGui::DragFloat("+Y##wd", &settings.maxY, 1.f, 0.f, 2000.f, "%.0f");
        ImGui::SetItemTooltip("Ceiling Y.");
        ImGui::TableSetColumnIndex(1);
        bchg |= ImGui::DragFloat("-Y##wd", &settings.minY, 1.f, -2000.f, 0.f, "%.0f");
        ImGui::SetItemTooltip("Floor Y.");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        bchg |= ImGui::DragFloat("+Z##wd", &settings.maxz, 1.f, 0.f, 2000.f, "%.0f");
        ImGui::SetItemTooltip("Back wall Z.");
        ImGui::TableSetColumnIndex(1);
        bchg |= ImGui::DragFloat("-Z##wd", &settings.minZ, 1.f, -2000.f, 0.f, "%.0f");
        ImGui::SetItemTooltip("Front wall Z.");

        ImGui::EndTable();
    }
    if (bchg) { initBoundingBox(); syncsettings = true; }

    ImGui::Checkbox("Show bounding box##wd", &settings.boundingBox); SYNC;

    float bx = settings.maxX - settings.minX;
    float by = settings.maxY - settings.minY;
    float bz = settings.maxz - settings.minZ;
    ImGui::TextDisabled("%.0f x %.0f x %.0f   vol %.0f", bx, by, bz, bx * by * bz);

    // ── Time ─────────────────────────────────────────────────────────────────
    Sec("Time");
    ImGui::TextDisabled("Speed / Substeps are in the Quick tab.");
    ImGui::Spacing();
    ImGui::InputFloat("Fixed dt##wd", &settings.fixedDt, 0.f, 0.f, "%.6f"); SYNC;
    ImGui::SetItemTooltip(
        "Physics timestep in seconds.\n"
        "Default 1/120 = 0.008333 s.\n"
        "Smaller = more accurate but heavier.");
}

// ─── RENDER ──────────────────────────────────────────────────────────────────
static void DrawRenderContent()
{
    // ── Mode ─────────────────────────────────────────────────────────────────
    Sec("Mode");
    {
        static const char* modeNames[] = {
            "0  Screen-Space Water",
            "1  Particles  (lit spheres)"
        };
        if (ImGui::Combo("##rnmode", &settings.shaderType, modeNames, 2)) syncsettings = true;
        ImGui::SetItemTooltip(
            "0 = Screen-space fluid surface with sky reflection,\n"
            "    Beer-Lambert absorption, two-lobe specular.\n"
            "    Heat effect invisible (under fluid surface).\n\n"
            "1 = Classic lit sphere particles.\n"
            "    Heat colour effect available.\n"
            "    Much faster — use for high particle counts.");
    }

    // ── Water — mode 0 only ───────────────────────────────────────────────────
    if (settings.shaderType == 0)
    {
        Sec("Water Colour");
        if (ImGui::ColorEdit3("Shallow##rn", &settings.shallowColorR)) syncsettings = true;
        ImGui::SetItemTooltip("Colour at thin regions / fluid surface.  Visible where particles barely overlap.");
        if (ImGui::ColorEdit3("Deep##rn", &settings.deepColorR))    syncsettings = true;
        ImGui::SetItemTooltip("Colour deep inside the fluid body.  Beer-Lambert absorption drives the transition.");

        Sec("Absorption");
        ImGui::SliderFloat("Coeff##rnabs", &settings.absorption, 0.1f, 8.0f, "%.2f"); SYNC;
        ImGui::SetItemTooltip(
            "Beer-Lambert absorption coefficient.\n"
            "0.3 = crystal-clear.   1.4 = default water.   5.0 = murky / deep ocean.");

        Sec("Sky Reflection");
        if (ImGui::ColorEdit3("Zenith##rn", &settings.skyZenithR))  syncsettings = true;
        ImGui::SetItemTooltip("Sky colour directly overhead.  Sampled when reflected ray points up.");
        if (ImGui::ColorEdit3("Horizon##rn", &settings.skyHorizonR)) syncsettings = true;
        ImGui::SetItemTooltip("Sky colour at horizon.  Sampled when reflected ray is nearly horizontal.");
        ImGui::SliderFloat("Strength##rnrs", &settings.reflStrength, 0.0f, 1.5f, "%.2f"); SYNC;
        ImGui::SetItemTooltip(
            "Sky reflection intensity.\n"
            "1.0 = physically correct.   >1.0 = exaggerated bright sky.   0.0 = no sky reflection.");

        Sec("Surface Blur");
        ImGui::SliderFloat("Sigma (px)##rnbs", &settings.blurSigma, 1.0f, 14.0f, "%.1f"); SYNC;
        ImGui::SetItemTooltip(
            "Gaussian sigma in pixels for bilateral blur.\n"
            "Higher = particles merge into a smoother surface.\n"
            "Lower = individual particles more distinct.  2 H+V iterations, radius 8.");
        ImGui::SliderFloat("Edge hold##rnbd", &settings.blurDepthFall, 2.0f, 60.0f, "%.1f"); SYNC;
        ImGui::SetItemTooltip(
            "Bilateral depth-edge sharpness.\n"
            "Low = blurs freely across depth boundaries.\n"
            "High = hard surface edges.  Good range: 15-30.");
    }

    // ── Background — always visible ────────────────────────────────────────────
    Sec("Background");
    if (ImGui::ColorEdit3("BG##rn", &settings.bgColorR)) syncsettings = true;
    ImGui::SetItemTooltip("Clear colour behind the simulation.  Dark blue-grey complements water in mode 0.");
}

// ─── PERF ────────────────────────────────────────────────────────────────────
static void DrawPerfContent()
{
    Sec("Frame Time");

    {
        float  f = settings.avgFps;
        float  n = UIClamp(f / 165.f, 0.f, 1.f);
        ImVec4 col = f > 100 ? ImVec4(0.28f, 0.58f, 0.44f, 1)
            : f > 50 ? ImVec4(0.62f, 0.50f, 0.20f, 1)
            : ImVec4(0.58f, 0.28f, 0.28f, 1);
        char lbl[32]; sprintf(lbl, "%.0f fps", f);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, col);
        ImGui::ProgressBar(n, ImVec2(-1, 12), lbl);
        ImGui::PopStyleColor();
        ImGui::SetItemTooltip("Average FPS.  Teal > 100.  Amber > 50.  Red <= 50.");
    }

    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.08f, 0.08f, 0.08f, 1));
    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.52f, 0.64f, 0.74f, 1));
    ImGui::PlotLines("##pfring", s_ring, IM_ARRAYSIZE(s_ring), s_head,
        nullptr, 0.f, 200.f, ImVec2(ImGui::GetContentRegionAvail().x, 44));
    ImGui::PopStyleColor(2);
    ImGui::SetItemTooltip("FPS — last 128 frames.");

    ImGui::Spacing();
    if (BeginStat2("##pffst"))
    {
        StatRow("Avg", "%.1f ms / %.0f fps",
            1000.f / UIMax(settings.avgFps, 0.1f), settings.avgFps);
        StatRow("Min", "%.0f fps", settings.minFps);
        StatRow("Max", "%.0f fps", settings.maxFps);
        ImGui::EndTable();
    }

    Sec("Simulation");
    if (BeginStat2("##pfsst"))
    {
        StatRow("Active", "%d", settings.count);
        StatRow("Capacity", "%d", settings.maxparticles);
        StatRow("Substeps", "%d", settings.substeps);
        StatRow("Speed", "%.2fx", settings.simspeed);
        StatRow("dt", "%.5f s", settings.fixedDt);
        ImGui::EndTable();
    }

    ImGui::Spacing();
    FillBar(6.f);

    Sec("Pipeline  (rough complexity)");
    ImGui::TextDisabled("computeHash      O(N)");
    ImGui::TextDisabled("radix sort       O(N log N)");
    ImGui::TextDisabled("reorder          O(N)");
    ImGui::TextDisabled("computeDensity   O(N x K)");
    ImGui::TextDisabled("computePressure  O(N x K)");
    ImGui::TextDisabled("updateKernel     O(N)");
    ImGui::TextDisabled("packVBO          O(N)");
}

// ─── HELP ────────────────────────────────────────────────────────────────────
static void DrawHelpContent()
{
    Sec("Camera");
    if (BeginStat2("##hlcam"))
    {
        StatRow("W / S", "forward / back");
        StatRow("A / D", "strafe left / right");
        StatRow("Q / E", "move up / down");
        StatRow("Mouse", "look around");
        ImGui::EndTable();
    }

    Sec("Controls");
    if (BeginStat2("##hlctrl"))
    {
        StatRow("Space", "pause / resume");
        StatRow("K", "restart simulation");
        ImGui::EndTable();
    }

    Sec("SPH Parameter Guide");

    auto Tip = [](const char* name, const char* body) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.72f, 0.72f, 0.70f, 1), "%s", name);
        ImGui::Indent(10);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.46f, 0.46f, 0.44f, 1));
        ImGui::TextWrapped("%s", body);
        ImGui::PopStyleColor();
        ImGui::Unindent(10);
        };

    Tip("Performance tip",
        "For maximum FPS, switch to Particle (mode 1) rendering in the Render tab.  "
        "Screen-space water requires multiple render passes.");
    Tip("h — smoothing radius",
        "Master kernel scale.  Larger = more neighbours = thicker fluid.  "
        "All coefficients recomputed on change.");
    Tip("Rest density",
        "Equilibrium target.  Raise to compact particles; lower to spread them.");
    Tip("Stiffness k",
        "Compression resistance.  Too high = instability.  "
        "Start low (~100-500), increase gradually while watching substeps.");
    Tip("Near k'",
        "Short-range repulsion.  Raise when particles clump.  Typically 10-50x k.");
    Tip("Viscosity",
        "Velocity averaging between neighbours.  Low = water.  High = honey.");
    Tip("Substeps",
        "Stability at high k.  2-4 recommended.  Linear GPU cost.");
    Tip("Wall force / dist",
        "Soft boundary repulsion inside the bounding box.  Tune if particles "
        "escape or cluster near walls.");

    Sec("About");
    ImGui::TextDisabled("3D SPH — CUDA + OpenGL");
    ImGui::TextDisabled("Muller poly6 / spiky kernels");
    ImGui::TextDisabled("Symplectic Euler integration");
    ImGui::TextDisabled("CUDA / GL VBO interop");
    ImGui::TextDisabled("Spatial hash  ->  radix sort  ->  reorder");
}

// ═════════════════════════════════════════════════════════════════════════════
//  ui_init  —  called every frame
// ═════════════════════════════════════════════════════════════════════════════
void ui_init()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ApplyTheme();

    // FPS ring update
    s_ring[s_head] = settings.fps;
    s_head = (s_head + 1) % IM_ARRAYSIZE(s_ring);

    ImGuiIO& io = ImGui::GetIO();
    const float SW = io.DisplaySize.x;
    const float SH = io.DisplaySize.y;

    // ── HUD  — top-left corner ────────────────────────────────────────────────
    {
        ImDrawList* dl = ImGui::GetForegroundDrawList();
        const float lh = ImGui::GetTextLineHeight() + 3.0f;
        const float x = 14.0f;
        float       y = 10.0f;
        char b[128];

        ImU32 fpsCol = settings.avgFps > 100 ? IM_COL32(96, 188, 148, 210)
            : settings.avgFps > 50 ? IM_COL32(196, 172, 84, 210)
            : IM_COL32(188, 100, 100, 230);
        sprintf(b, "%.0f fps   min %.0f   max %.0f",
            settings.avgFps, settings.minFps, settings.maxFps);
        dl->AddText(ImVec2(x, y), fpsCol, b); y += lh;

        sprintf(b, "%d / %d particles", settings.count, settings.maxparticles);
        dl->AddText(ImVec2(x, y), IM_COL32(148, 168, 196, 180), b); y += lh;

        sprintf(b, "cam  %.1f  %.1f  %.1f", settings.wx, settings.wy, settings.wz);
        dl->AddText(ImVec2(x, y), IM_COL32(120, 180, 140, 155), b); y += lh;

        sprintf(b, "substeps %d   speed %.2fx", settings.substeps, settings.simspeed);
        dl->AddText(ImVec2(x, y), IM_COL32(120, 120, 118, 148), b); y += lh;

        if (!settings.nopause)
            dl->AddText(ImVec2(x, y), IM_COL32(188, 100, 100, 230), "PAUSED  --  Space to resume");
        else
            dl->AddText(ImVec2(x, y), IM_COL32(96, 168, 128, 170), "running");
    }

    // ── Floating detached-tab windows ─────────────────────────────────────────
    //  Rendered BEFORE the sidebar so they sit on top when overlapping.
    typedef void (*DrawFn)();
    static const DrawFn kDraw[TAB_COUNT] = {
        DrawQuickContent, DrawFluidContent, DrawParticlesContent, DrawWorldContent,
        DrawRenderContent, DrawPerfContent, DrawHelpContent
    };

    for (int i = 0; i < TAB_COUNT; i++)
    {
        if (!s_det[i]) continue;
        PrepareFloatWindow(i);
        if (ImGui::Begin(kTab[i], nullptr,
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoNav))
        {
            PinButton(i);
            ImGui::BeginChild("##fc", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);
            kDraw[i]();
            ImGui::EndChild();
        }
        ImGui::End();
    }

    // ── Main sidebar panel ────────────────────────────────────────────────────
    ImGui::SetNextWindowPos(ImVec2(SW - PANEL_W, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(PANEL_W, SH), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.97f);

    ImGuiWindowFlags wf =
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoTitleBar;

    ImGui::Begin("##panel", nullptr, wf);

    // Status badge strip
    {
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
        ImGui::TextDisabled("SPH FLUID");
        ImGui::SameLine();

        float bx = PANEL_W - 10.0f;
        if (settings.addParticle) {
            bx -= 72; ImGui::SetCursorPosX(bx);
            Badge(" EMIT ", ImVec4(0.18f, 0.32f, 0.44f, 1));
            ImGui::SameLine();
        }
        if (!settings.colisionFun) {
            bx -= 74; ImGui::SetCursorPosX(bx);
            Badge(" SPH OFF ", ImVec4(0.38f, 0.30f, 0.10f, 1));
            ImGui::SameLine();
        }
        if (!settings.nopause) {
            bx -= 62; ImGui::SetCursorPosX(bx);
            Badge(" PAUSED ", ImVec4(0.38f, 0.16f, 0.16f, 1));
        }
    }

    ImGui::Separator();

    // ── Tab bar — detached tabs are hidden ────────────────────────────────────
    bool anyDocked = false;
    for (int i = 0; i < TAB_COUNT; i++) if (!s_det[i]) { anyDocked = true; break; }

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 5));

    if (anyDocked &&
        ImGui::BeginTabBar("##tabs", ImGuiTabBarFlags_NoCloseWithMiddleMouseButton))
    {
        ImGui::PopStyleVar();

        // Helper lambda: wraps a tab item + scroll child + detach button + content
        // Using a local macro to avoid repeating the boilerplate 7 times.
        // (C++ lambdas can't easily take a void(*)() param cleanly here,
        //  so we spell out each tab — compiler will inline anyway.)

#define DRAW_TAB(ID, FN) \
        if (!s_det[ID] && ImGui::BeginTabItem(kTab[ID])) { \
            ImGui::BeginChild("##c"#ID, ImVec2(0,0), false, \
                ImGuiWindowFlags_HorizontalScrollbar); \
            DetachButton(ID); \
            FN(); \
            ImGui::EndChild(); \
            ImGui::EndTabItem(); \
        }

        DRAW_TAB(TAB_QUICK, DrawQuickContent)
            DRAW_TAB(TAB_FLUID, DrawFluidContent)
            DRAW_TAB(TAB_PARTICLES, DrawParticlesContent)
            DRAW_TAB(TAB_WORLD, DrawWorldContent)
            DRAW_TAB(TAB_RENDER, DrawRenderContent)
            DRAW_TAB(TAB_PERF, DrawPerfContent)
            DRAW_TAB(TAB_HELP, DrawHelpContent)

#undef DRAW_TAB

            ImGui::EndTabBar();
    }
    else
    {
        ImGui::PopStyleVar();
        if (!anyDocked) {
            ImGui::Spacing();
            ImGui::TextDisabled("All tabs are floating.");
            ImGui::TextDisabled("Click [pin] in any floating window to restore it here.");
        }
    }

    ImGui::End();  // ##panel

  
}