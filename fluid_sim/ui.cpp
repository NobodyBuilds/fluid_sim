#include<iostream>
#include "ui.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include<algorithm>
#include "settings.h"
#include "main.h"

static inline float UIClamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float UIMax(float a, float b) { return a > b ? a : b; }

// ─────────────────────────────────────────────────────────────────────────────
//  Panel geometry  — change PANEL_W to widen / narrow the sidebar
// ─────────────────────────────────────────────────────────────────────────────
static const float PANEL_W = 340.0f;

// ─────────────────────────────────────────────────────────────────────────────
//  Theme  — boxy, compact, calm mid-grey
//  Palette:
//    bg0  #1a1a1a   window / panel body
//    bg1  #222222   child regions
//    bg2  #2c2c2c   frames / inputs
//    bg3  #363636   hovered frames
//    bg4  #424242   active frames
//    ln   #3a3a3a   borders / separators
//    txt  #c8c8c4   primary text
//    dim  #686864   disabled / hints
//    acc  #8a9aa8   accent (muted steel-blue for sliders / checks)
// ─────────────────────────────────────────────────────────────────────────────
static void ApplyTheme()
{
    ImGuiStyle& s = ImGui::GetStyle();

    // Boxy — minimal rounding
    s.WindowRounding = 0.0f;
    s.ChildRounding = 0.0f;
    s.FrameRounding = 2.0f;
    s.PopupRounding = 2.0f;
    s.ScrollbarRounding = 2.0f;
    s.GrabRounding = 2.0f;
    s.TabRounding = 2.0f;

    // Thin / compact spacing
    s.WindowPadding = ImVec2(10, 8);
    s.FramePadding = ImVec2(6, 3);
    s.ItemSpacing = ImVec2(6, 4);
    s.ItemInnerSpacing = ImVec2(4, 3);
    s.IndentSpacing = 14.0f;
    s.ScrollbarSize = 9.0f;
    s.GrabMinSize = 8.0f;
    s.WindowBorderSize = 1.0f;
    s.FrameBorderSize = 1.0f;
    s.TabBorderSize = 0.0f;
    s.SeparatorTextBorderSize = 1.0f;
    s.SeparatorTextPadding = ImVec2(6, 2);

    // Lightened palette — base lifted from #191919 to #2e2e2e range
    //   bg0  #2e2e2e   window body
    //   bg1  #363636   child regions
    //   bg2  #404040   frames / inputs
    //   bg3  #4a4a4a   hovered frames
    //   bg4  #565656   active frames
    //   ln   #525252   borders / separators  (clearly visible)
    //   txt  #deded8   primary text  (brighter)
    //   dim  #888882   disabled / hints  (readable)
    //   acc  #7aacc4   muted steel-blue accent
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
    // Accent — muted steel blue
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

// Calm inline badge — uses muted tones only
static void Badge(const char* label, ImVec4 col)
{
    ImGui::PushStyleColor(ImGuiCol_Button, col);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, col);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, col);
    ImGui::SmallButton(label);
    ImGui::PopStyleColor(3);
}

// Full-width muted-red button
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
//  FPS ring buffer
// ─────────────────────────────────────────────────────────────────────────────
static float s_ring[128] = {};
static int   s_head = 0;

// ─────────────────────────────────────────────────────────────────────────────
//  ui_init  — call every frame
// ─────────────────────────────────────────────────────────────────────────────
void ui_init()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ApplyTheme();

    s_ring[s_head] = settings.fps;
    s_head = (s_head + 1) % IM_ARRAYSIZE(s_ring);

    ImGuiIO& io = ImGui::GetIO();
    const float SW = io.DisplaySize.x;   // screen width
    const float SH = io.DisplaySize.y;   // screen height

    // ── HUD  — top-left corner, always on top ─────────────────────────────────
    {
        ImDrawList* dl = ImGui::GetForegroundDrawList();
        const float lh = ImGui::GetTextLineHeight() + 3.0f;
        const float x = 14.0f;
        float       y = 10.0f;
        char b[128];

        // FPS — calm colours: teal / warm amber / dusty rose
        ImU32 fpsCol = settings.avgFps > 100 ? IM_COL32(96, 188, 148, 210)
            : settings.avgFps > 50 ? IM_COL32(196, 172, 84, 210)
            : IM_COL32(188, 100, 100, 230);
        sprintf(b, "%.0f fps   min %.0f   max %.0f",
            settings.avgFps, settings.minFps, settings.maxFps);
        dl->AddText(ImVec2(x, y), fpsCol, b);                            y += lh;

        sprintf(b, "%d / %d particles", settings.count, settings.maxparticles);
        dl->AddText(ImVec2(x, y), IM_COL32(148, 168, 196, 180), b);         y += lh;

        sprintf(b, "cam  %.1f  %.1f  %.1f", settings.wx, settings.wy, settings.wz);
        dl->AddText(ImVec2(x, y), IM_COL32(120, 180, 140, 155), b);         y += lh;

        sprintf(b, "substeps %d   speed %.2fx", settings.substeps, settings.simspeed);
        dl->AddText(ImVec2(x, y), IM_COL32(120, 120, 118, 148), b);         y += lh;

        // Pause / running  — driven by settings.nopause
        if (!settings.nopause)
            dl->AddText(ImVec2(x, y), IM_COL32(188, 100, 100, 230),
                "PAUSED  —  Space to resume");
        else
            dl->AddText(ImVec2(x, y), IM_COL32(96, 168, 128, 170),
                "running");
    }

    // ── Right-side panel  — full height, fixed ────────────────────────────────
    ImGui::SetNextWindowPos(ImVec2(SW - PANEL_W, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(PANEL_W, SH), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.97f);

    ImGuiWindowFlags wf =
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoTitleBar;          // clean — no title, just tabs

    ImGui::Begin("##panel", nullptr, wf);

    // Thin header strip with status badges
    {
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
        ImGui::TextDisabled("SPH FLUID");
        ImGui::SameLine();

        // Push badges to right side of header
        float bx = PANEL_W - 10.0f;
        if (settings.addParticle) {
            bx -= 72; ImGui::SetCursorPosX(bx);
            Badge(" EMIT ", ImVec4(0.18f, 0.32f, 0.44f, 1));
            ImGui::SameLine();
        }
        if (!settings.colisionFun) {
            bx -= 66; ImGui::SetCursorPosX(bx);
            Badge(" SPH OFF ", ImVec4(0.38f, 0.30f, 0.10f, 1));
            ImGui::SameLine();
        }
        if (!settings.nopause) {
            bx -= 58; ImGui::SetCursorPosX(bx);
            Badge(" PAUSED ", ImVec4(0.38f, 0.16f, 0.16f, 1));
        }
    }

    ImGui::Separator();

    // ── Tab bar ───────────────────────────────────────────────────────────────
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 5));
    if (ImGui::BeginTabBar("tabs", ImGuiTabBarFlags_NoCloseWithMiddleMouseButton))
    {
        ImGui::PopStyleVar();

        // ══════════════════════════════════════════════════════════════════════
        //  QUICK
        // ══════════════════════════════════════════════════════════════════════
        if (ImGui::BeginTabItem("Quick"))
        {
            ImGui::BeginChild("##q", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

            Sec("Simulation");

            // Pause / resume button mirroring Space key
            {
                bool paused = !settings.nopause;
                const char* lbl = paused ? "Resume  (Space)" : "Pause  (Space)";
                ImVec4 bc = paused ? ImVec4(0.38f, 0.16f, 0.16f, 1) : ImVec4(0.20f, 0.20f, 0.20f, 1);
                ImVec4 bh = paused ? ImVec4(0.50f, 0.22f, 0.22f, 1) : ImVec4(0.28f, 0.28f, 0.28f, 1);
                ImGui::PushStyleColor(ImGuiCol_Button, bc);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bh);
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.44f, 0.44f, 0.44f, 1));
                if (ImGui::Button(lbl, ImVec2(-1, 0)))
                    settings.nopause = !settings.nopause;
                ImGui::PopStyleColor(3);
                ImGui::SetItemTooltip(
                    "Toggle pause / resume.\n"
                    "Same as pressing Space.");
            }

            ImGui::Spacing();
            ImGui::SliderFloat("Speed##q", &settings.simspeed, 0.001f, 10.0f, "%.2f");
            ImGui::SetItemTooltip(
                "Multiplier on the physics timestep.\n"
                "1.0 = real time.   0.2 = slow motion.   3.0 = fast forward.");

            ImGui::InputInt("Substeps##q", &settings.substeps);
            if (settings.substeps < 1) settings.substeps = 1;
            ImGui::SetItemTooltip(
                "Physics steps per rendered frame.\n"
                "Higher = more stable at large stiffness,\nbut linearly more GPU cost.");

            Sec("Particles");

            ImGui::InputInt("Count##q", &settings.totalBodies);
            if (ImGui::IsItemDeactivatedAfterEdit()) restartSimulation();
            ImGui::SetItemTooltip(
                "Particles spawned on restart.\nPress Enter to confirm — restarts automatically.");

            ImGui::InputInt("Buffer##q", &settings.maxparticles);
            if (ImGui::IsItemDeactivatedAfterEdit()) restartSimulation();
            ImGui::SetItemTooltip(
                "GPU allocation size in particles.\nMust be >= Count.");

            {
                float fill = settings.maxparticles > 0
                    ? (float)settings.count / (float)settings.maxparticles : 0.f;
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                    fill > 0.88f ? ImVec4(0.58f, 0.24f, 0.24f, 1)
                    : ImVec4(0.34f, 0.42f, 0.50f, 1));
                ImGui::ProgressBar(fill, ImVec2(-1, 4), "");
                ImGui::PopStyleColor();
                ImGui::SetItemTooltip("Buffer fill  %d / %d  (%.0f%%)",
                    settings.count, settings.maxparticles, fill * 100.f);
            }

            Sec("Key Physics");

            ImGui::SliderFloat("Gravity##q", &settings.downf, 0.0f, 1000.0f, "%.0f");
            ImGui::SetItemTooltip(
                "Constant downward acceleration per step.\n0 = weightless.   150 = default.");

            if (ImGui::SliderFloat("h##q", &settings.h, 0.1f, 20.0f, "%.2f"))
                calcKernels();
            ImGui::SetItemTooltip(
                "Smoothing radius — particles within this distance interact.\n"
                "Larger h = thicker, slower fluid.\n"
                "All kernel coefficients update automatically.");

            ImGui::DragFloat("Rest rho##q", &settings.rest_density, 0.5f, 0.f, 10000.f, "%.1f");
            ImGui::SetItemTooltip(
                "Target equilibrium density.\n"
                "Higher = particles draw together.\nLower = fluid expands.");

            Sec("Toggles");

            ImGui::Checkbox("SPH##q", &settings.colisionFun);
            ImGui::SetItemTooltip("Enable density + pressure force kernels.");
            ImGui::SameLine(120);
            ImGui::Checkbox("Integrate##q", &settings.updateFun);
            ImGui::SetItemTooltip("Enable velocity / position update step.");
            ImGui::SameLine();
            ImGui::Checkbox("Heat##q", &settings.heateffect);
            ImGui::SetItemTooltip("Enable kinetic heat colour shift.");

            ImGui::Spacing();
            if (DangerButton("Restart"))
                restartSimulation();
            ImGui::SetItemTooltip("Wipe all particles and respawn.");

            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        // ══════════════════════════════════════════════════════════════════════
        //  FLUID
        // ══════════════════════════════════════════════════════════════════════
        if (ImGui::BeginTabItem("Fluid"))
        {
            ImGui::BeginChild("##fl", ImVec2(0, 0), false);

            Sec("Kernel");

            if (ImGui::SliderFloat("h##fl", &settings.h, 0.1f, 20.0f, "%.2f"))
                calcKernels();
            ImGui::SetItemTooltip(
                "Interaction radius.\n"
                "All kernel coefficients derived from this via calcKernels().\n"
                "Increase for thicker / more cohesive fluid.");

            if (ImGui::TreeNodeEx("Coefficients  (read only)",
                ImGuiTreeNodeFlags_SpanFullWidth))
            {
                ImGui::BeginDisabled();
                ImGui::InputFloat("poly6", &settings.pollycoef6, 0, 0, "%.4e");
                ImGui::InputFloat("spiky", &settings.spikycoef, 0, 0, "%.4e");
                ImGui::InputFloat("spiky grad", &settings.spikygradv, 0, 0, "%.4e");
                ImGui::InputFloat("visc", &settings.viscosity, 0, 0, "%.4e");
                ImGui::InputFloat("self rho", &settings.Sdensity, 0, 0, "%.4e");
                ImGui::InputFloat("near self", &settings.ndensity, 0, 0, "%.4e");
                ImGui::EndDisabled();
                ImGui::TreePop();
            }

            Sec("Density");

            ImGui::DragFloat("Rest rho", &settings.rest_density, 0.5f, 0.f, 10000.f, "%.2f");
            ImGui::SetItemTooltip(
                "Target density at equilibrium.\n"
                "Raise to attract particles.  Lower to spread them.");

            Sec("Pressure");

            ImGui::DragFloat("Stiffness k", &settings.pressure, 1.f, 0.f, 200000.f, "%.0f");
            ImGui::SetItemTooltip(
                "Compression resistance.\n"
                "High = nearly incompressible but may cause instability.\n"
                "Start ~100, increase gradually.");

            ImGui::DragFloat("Near k'", &settings.nearpressure, 1.f, 0.f, 200000.f, "%.0f");
            ImGui::SetItemTooltip(
                "Short-range repulsion.\n"
                "Prevents collapse at very close range.\n"
                "Typically 10–50× the value of k.");

            Sec("Viscosity");

            ImGui::DragFloat("Viscosity", &settings.visc, 0.01f, 0.f, 200.f, "%.3f");
            ImGui::SetItemTooltip(
                "Velocity averaging between neighbours.\n"
                "Low = water.   High = honey / thick fluid.");

            Sec("Forces");

            ImGui::SliderFloat("Gravity", &settings.downf, 0.f, 1000.f, "%.0f");
            ImGui::SetItemTooltip("Downward acceleration each step.");

            ImGui::InputFloat("Restitution", &settings.restitution, 0.02f, 0.1f, "%.3f");
            ImGui::SetItemTooltip(
                "Wall bounce coefficient.\n"
                "0.0 = fully inelastic.   1.0 = perfectly elastic.");

            Sec("Pipeline");

            ImGui::Checkbox("SPH forces", &settings.colisionFun);
            ImGui::SameLine(150);
            ImGui::Checkbox("Integration", &settings.updateFun);

            Sec("additonal settings");
            ImGui::Text("Pressure accumulation mode:");
            ImGui::SetItemTooltip(
                "SPH pressure can be accumulated in two ways:\n"
                "- sign:   pressure from neighbours is subtracted from self density.\n"
                "+ sign:   pressure from neighbours is added to self density.\n"
                "The - sign method is more common and slightly faster, but the + sign method can produce more stable results at high stiffness with fewer substeps.");

            ImGui::RadioButton("- sign pressure accumulation", &settings.pressureMode, 0);
            ImGui::RadioButton("+ sign pressure accumulation", &settings.pressureMode, 1);

            ImGui::Spacing();
            ImGui::Checkbox("pressure clamping", &settings.pressureClamp);
            ImGui::SetItemTooltip(
                "Clamp pressure contribution from neighbours.\n"
                "prevents any negative pressure but may cause bad behaviour "
            );
            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        // ══════════════════════════════════════════════════════════════════════
        //  PARTICLES
        // ══════════════════════════════════════════════════════════════════════
        if (ImGui::BeginTabItem("Particles"))
        {
            ImGui::BeginChild("##pt", ImVec2(0, 0), false);

            Sec("Spawn");

            ImGui::InputInt("Count", &settings.totalBodies);
            if (ImGui::IsItemDeactivatedAfterEdit()) restartSimulation();
            ImGui::SetItemTooltip("Particles on next restart.  Enter to confirm.");

            ImGui::InputInt("Buffer", &settings.maxparticles);
            if (ImGui::IsItemDeactivatedAfterEdit()) restartSimulation();
            ImGui::SetItemTooltip(
                "GPU buffer size.  Must be >= Count.\n"
                "Increase before using the emitter.");

            {
                float fill = settings.maxparticles > 0
                    ? (float)settings.count / (float)settings.maxparticles : 0.f;
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                    fill > 0.88f ? ImVec4(0.58f, 0.24f, 0.24f, 1)
                    : ImVec4(0.34f, 0.42f, 0.50f, 1));
                ImGui::ProgressBar(fill, ImVec2(-1, 4), "");
                ImGui::PopStyleColor();
                ImGui::SetItemTooltip("Fill  %d / %d  (%.0f%%)",
                    settings.count, settings.maxparticles, fill * 100.f);
            }

            ImGui::Spacing();
            ImGui::InputFloat("Radius", &settings.size, 0.05f, 0.2f, "%.3f");
            if (ImGui::IsItemDeactivatedAfterEdit()) restartSimulation();
            ImGui::SetItemTooltip("Visual radius.  Restart required.");

            ImGui::InputFloat("Mass", &settings.particleMass, 0.1f, 0.5f, "%.3f");
            if (ImGui::IsItemDeactivatedAfterEdit()) restartSimulation();
            ImGui::SetItemTooltip("Particle mass.  Affects pressure / density contributions.");

            Sec("Spawn Tint");

            ImGui::SliderInt("R", &settings.rc, 0, 255);
            ImGui::SliderInt("G", &settings.gc, 0, 255);
            ImGui::SliderInt("B", &settings.bc, 0, 255);
            ImGui::ColorButton("##sw",
                ImVec4(settings.rc / 255.f, settings.gc / 255.f, settings.bc / 255.f, 1.f),
                ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoBorder,
                ImVec2(ImGui::GetContentRegionAvail().x, 14));

            Sec("Heat Colour");

            ImGui::Checkbox("Enable", &settings.heateffect);
            ImGui::SetItemTooltip(
                "Shift particle colour by kinetic energy.\n"
                "Blue = slow   Red = fast.");
            if (settings.heateffect)
            {
                ImGui::SliderFloat("Gen", &settings.heatMultiplier, 0.1f, 20.f, "%.1f");
                ImGui::SetItemTooltip("How quickly movement raises the heat colour.");
                ImGui::SliderFloat("Fade", &settings.cold, 0.1f, 20.f, "%.1f");
                ImGui::SetItemTooltip("Decay rate — how fast colour cools when idle.");
            }

            Sec("Emitter");

            ImGui::Checkbox("Enable##em", &settings.addParticle);
            ImGui::SetItemTooltip("Inject new particles each frame.");
            if (settings.addParticle)
            {
                ImGui::InputInt("Flow", &settings.flowcount);
                ImGui::SetItemTooltip(
                    "Particles injected per frame.\n"
                    "Keep low (1–20) to avoid sudden overflow.");
                ImGui::TextDisabled("emitted  %d", settings.samplecount);
            }

            ImGui::Spacing();
            if (DangerButton("Restart"))
                restartSimulation();

            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        // ══════════════════════════════════════════════════════════════════════
        //  WORLD
        // ══════════════════════════════════════════════════════════════════════
        if (ImGui::BeginTabItem("World"))
        {
            ImGui::BeginChild("##wd", ImVec2(0, 0), false);

            Sec("Bounding Box");
            ImGui::TextDisabled("Drag — updates in real time.");
            ImGui::Spacing();

            bool bchg = false;
            if (ImGui::BeginTable("##bb", 2, ImGuiTableFlags_SizingStretchSame))
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                bchg |= ImGui::DragFloat("+X", &settings.maxX, 1.f, 0.f, 2000.f, "%.0f");
                ImGui::SetItemTooltip("Right wall X.  Drag left to shrink.");
                ImGui::TableSetColumnIndex(1);
                bchg |= ImGui::DragFloat("-X", &settings.minX, 1.f, -2000.f, 0.f, "%.0f");
                ImGui::SetItemTooltip("Left wall X.  Drag right to shrink.");

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                bchg |= ImGui::DragFloat("+Y", &settings.maxY, 1.f, 0.f, 2000.f, "%.0f");
                ImGui::TableSetColumnIndex(1);
                bchg |= ImGui::DragFloat("-Y", &settings.minY, 1.f, -2000.f, 0.f, "%.0f");

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                bchg |= ImGui::DragFloat("+Z", &settings.maxz, 1.f, 0.f, 2000.f, "%.0f");
                ImGui::SetItemTooltip("Ceiling Z.");
                ImGui::TableSetColumnIndex(1);
                bchg |= ImGui::DragFloat("-Z", &settings.minZ, 1.f, -2000.f, 0.f, "%.0f");
                ImGui::SetItemTooltip("Floor Z.");

                ImGui::EndTable();
            }
            if (bchg) initBoundingBox();

            float bx = settings.maxX - settings.minX;
            float by = settings.maxY - settings.minY;
            float bz = settings.maxz - settings.minZ;
            ImGui::TextDisabled("%.0f x %.0f x %.0f   vol %.0f", bx, by, bz, bx * by * bz);

            Sec("Time");

            ImGui::SliderFloat("Speed", &settings.simspeed, 0.001f, 10.f, "%.3f");
            ImGui::SetItemTooltip(
                "dt multiplier.\n"
                "0.1 = slow motion.   1.0 = real time.   3.0 = fast forward.");

            ImGui::InputFloat("Fixed dt", &settings.fixedDt, 0.f, 0.f, "%.6f");
            ImGui::SetItemTooltip(
                "Physics timestep in seconds.\n"
                "Default 1/120 = 0.008333 s.\n"
                "Smaller = more accurate but heavier.");

            ImGui::InputInt("Substeps", &settings.substeps);
            if (settings.substeps < 1) settings.substeps = 1;
            ImGui::SetItemTooltip(
                "Steps per rendered frame.\n"
                "2–4 helps stability at high k values.\n"
                "Cost is linear — 4× substeps = ~4× GPU time.");

            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Render"))
        {
            ImGui::BeginChild("##rn", ImVec2(0, 0), false);

            // ── Mode selector ─────────────────────────────────────────────────
            Sec("Mode");
            {
                static const char* modeNames[] = {
                    "0  Screen-Space Water",
                    "1  Particles  (lit spheres)"
                };
                ImGui::Combo("##st", &settings.shaderType, modeNames, 2);
                ImGui::SetItemTooltip(
                    "0 = Screen-space fluid surface with sky reflection,\n"
                    "    Beer-Lambert absorption, two-lobe specular.\n"
                    "    Heat effect disabled (invisible under fluid surface).\n\n"
                    "1 = Classic lit sphere particles.\n"
                    "    Heat color effect available.");
            }

            // ── Water controls — shown only in mode 0 ─────────────────────────
            if (settings.shaderType == 0)
            {
                // ── Water colour ──────────────────────────────────────────────
                Sec("Water Color");
                ImGui::ColorEdit3("Shallow##sc", &settings.shallowColorR);
                ImGui::SetItemTooltip(
                    "Color at thin regions and the fluid surface.\n"
                    "Visible where particles barely overlap.");
                ImGui::ColorEdit3("Deep##dc", &settings.deepColorR);
                ImGui::SetItemTooltip(
                    "Color deep inside the fluid body.\n"
                    "Beer-Lambert absorption drives shallow→deep transition.");

                Sec("Absorption");
                ImGui::SliderFloat("Coeff##abs", &settings.absorption, 0.1f, 8.0f, "%.2f");
                ImGui::SetItemTooltip(
                    "Beer-Lambert absorption coefficient.\n"
                    "0.3 = crystal-clear.   1.4 = default water.\n"
                    "5.0 = murky / deep ocean.");

                // ── Sky reflection ────────────────────────────────────────────
                Sec("Sky Reflection");
                ImGui::ColorEdit3("Zenith##sz", &settings.skyZenithR);
                ImGui::SetItemTooltip(
                    "Sky color directly overhead.\n"
                    "Sampled when the reflected ray points upward.");
                ImGui::ColorEdit3("Horizon##sh", &settings.skyHorizonR);
                ImGui::SetItemTooltip(
                    "Sky color at the horizon.\n"
                    "Sampled when the reflected ray is nearly horizontal.");
                ImGui::SliderFloat("Strength##rs", &settings.reflStrength, 0.0f, 1.5f, "%.2f");
                ImGui::SetItemTooltip(
                    "Sky reflection intensity.\n"
                    "1.0 = physically correct.\n"
                    "> 1.0 = exaggerated bright sky.\n"
                    "0.0 = no sky reflection (just specular).");

                // ── Surface blur ──────────────────────────────────────────────
                Sec("Surface Blur");
                ImGui::SliderFloat("Sigma (px)##bs", &settings.blurSigma, 1.0f, 14.0f, "%.1f");
                ImGui::SetItemTooltip(
                    "Gaussian sigma in pixels for bilateral blur.\n"
                    "Higher = particles merge into a smoother surface.\n"
                    "Lower = individual particles more distinct.\n"
                    "2 H+V iterations (4 passes, radius 8).");
                ImGui::SliderFloat("Edge hold##bd", &settings.blurDepthFall, 2.0f, 60.0f, "%.1f");
                ImGui::SetItemTooltip(
                    "Bilateral depth-edge sharpness.\n"
                    "Low  = blurs freely across depth boundaries.\n"
                    "High = hard surface edges, less inter-particle merging.\n"
                    "Good range: 15–30.");
            }

            // ── Background — visible in both modes ────────────────────────────
            Sec("Background");
            ImGui::ColorEdit3("BG##bg", &settings.bgColorR);
            ImGui::SetItemTooltip(
                "Clear colour behind the simulation.\n"
                "Dark blue-grey complements water in mode 0.");

            ImGui::EndChild();
            ImGui::EndTabItem();
        }


        // ══════════════════════════════════════════════════════════════════════
        //  PERF
        // ══════════════════════════════════════════════════════════════════════
        if (ImGui::BeginTabItem("Perf"))
        {
            ImGui::BeginChild("##pf", ImVec2(0, 0), false);

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
                ImGui::SetItemTooltip(
                    "Average FPS.\n"
                    "Teal > 100.   Amber > 50.   Dusty rose <= 50.");
            }

            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.08f, 0.08f, 0.08f, 1));
            ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.52f, 0.64f, 0.74f, 1));
            ImGui::PlotLines("##sp", s_ring, IM_ARRAYSIZE(s_ring), s_head,
                nullptr, 0.f, 200.f,
                ImVec2(ImGui::GetContentRegionAvail().x, 44));
            ImGui::PopStyleColor(2);
            ImGui::SetItemTooltip("FPS — last 128 frames.");

            ImGui::Spacing();
            if (BeginStat2("##fst"))
            {
                StatRow("Avg",
                    "%.1f ms / %.0f fps",
                    1000.f / UIMax(settings.avgFps, 0.1f),
                    settings.avgFps);
                StatRow("Min", "%.0f fps", settings.minFps);
                StatRow("Max", "%.0f fps", settings.maxFps);
                ImGui::EndTable();
            }

            Sec("Simulation");

            if (BeginStat2("##sst"))
            {
                StatRow("Active", "%d", settings.count);
                StatRow("Capacity", "%d", settings.maxparticles);
                StatRow("Substeps", "%d", settings.substeps);
                StatRow("Speed", "%.2fx", settings.simspeed);
                StatRow("dt", "%.5f s", settings.fixedDt);
                ImGui::EndTable();
            }

            ImGui::Spacing();
            {
                float fill = settings.maxparticles > 0
                    ? (float)settings.count / (float)settings.maxparticles : 0.f;
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                    fill > 0.88f ? ImVec4(0.58f, 0.24f, 0.24f, 1)
                    : ImVec4(0.34f, 0.42f, 0.50f, 1));
                ImGui::ProgressBar(fill, ImVec2(-1, 6), "");
                ImGui::PopStyleColor();
                ImGui::SetItemTooltip("Buffer fill  %.1f%%", fill * 100.f);
            }

            Sec("Pipeline");
            ImGui::TextDisabled("computeHash      O(N)");
            ImGui::TextDisabled("radix sort       O(N log N)");
            ImGui::TextDisabled("reorder          O(N)");
            ImGui::TextDisabled("computeDensity   O(N x K)");
            ImGui::TextDisabled("computePressure  O(N x K)");
            ImGui::TextDisabled("updateKernel     O(N)");
            ImGui::TextDisabled("packVBO          O(N)");

            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        // ══════════════════════════════════════════════════════════════════════
        //  HELP
        // ══════════════════════════════════════════════════════════════════════
        if (ImGui::BeginTabItem("Help"))
        {
            ImGui::BeginChild("##hl", ImVec2(0, 0), false);

            Sec("Camera");
            if (BeginStat2("##cam"))
            {
                StatRow("W / S", "forward / back");
                StatRow("A / D", "strafe");
                StatRow("Q / E", "up / down");
                StatRow("Mouse", "look around");
                ImGui::EndTable();
            }

            Sec("Controls");
            if (BeginStat2("##ctrl"))
            {
                StatRow("Space", "pause / resume");
                StatRow("K", "restart");
                ImGui::EndTable();
            }

            Sec("SPH Parameters");

            auto Tip = [](const char* name, const char* body) {
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.72f, 0.72f, 0.70f, 1), "%s", name);
                ImGui::Indent(10);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.46f, 0.46f, 0.44f, 1));
                ImGui::TextWrapped("%s", body);
                ImGui::PopStyleColor();
                ImGui::Unindent(10);
                };

            Tip("perforamnce ",
                "for performance increase switch to particle only rendering mode in the render tab");

            Tip("h",
                "Master kernel scale.  Larger = more neighbours = "
                "thicker fluid.  All coefficients recomputed on change.");
            Tip("Rest density",
                "Equilibrium target.  Raise to compact particles; "
                "lower to spread them.");
            Tip("Stiffness k",
                "Compression resistance.  Too high = instability.  "
                "Start low, increase gradually.");
            Tip("Near k'",
                "Short-range repulsion.  Raise when particles clump.  "
                "Typically 10–50x k.");
            Tip("Viscosity",
                "Velocity averaging.  Low = water.  High = honey.");
            Tip("Substeps",
                "Stability at high k.  2–4 recommended.  "
                "Linear cost.");

            Sec("About");
            ImGui::TextDisabled("3D SPH — CUDA + OpenGL");
            ImGui::TextDisabled("Muller poly6 / spiky kernels");
            ImGui::TextDisabled("Symplectic Euler");
            ImGui::TextDisabled("CUDA / GL VBO interop");

            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();

    }
    ImGui::End();
}