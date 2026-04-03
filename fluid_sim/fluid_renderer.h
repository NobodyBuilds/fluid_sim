#pragma once
// ═══════════════════════════════════════════════════════════════════════════════
//  fluid_renderer.h  v4
//
//  shaderType 0 → Screen-space water    (this renderer — 4 passes)
//  shaderType 1 → Legacy particles      (render() returns false, caller draws)
//
//  Pipeline (mode 0):
//    Pass 1  Depth        sphere-surface depth → R32F FBO + gl_FragDepth
//    Pass 2  Thickness    additive chord accumulation (Beer-Lambert input)
//    Pass 3  Blur ×3      separable bilateral, radius 25, 3 H+V iterations (6 passes)
//    Pass 4  Composite    water shading: Fresnel, sky reflection, Beer-Lambert
//
//  v4 fixes (shaderType 0 only — shaderType 1 path UNCHANGED):
//    • SSF_BLUR_FRAG: RADIUS 12→25.  Bilateral depth sigma now uses a Gaussian
//      in world-scale units (sigmaD = uBlurDepthFall * 2.0) instead of a linear
//      exp(-|Δd|·falloff) that was zeroing out every tap at this scene scale.
//    • SSF_THICK_FRAG: chord coefficient 0.030→0.12.  Sparse particle density
//      (~2–3 overlap average at 30k/2.4M vol) produced near-zero thickness with
//      the old coefficient, making the fluid transparent even at max absorption.
//    • SSF_COMPOSITE_FRAG: thick clamp 3.0→15.0.  Alpha replaced with
//      Beer-Lambert coverage (1 − exp(−k·thick)) which is physically correct
//      and gives strong opacity at depth while keeping edges transparent.
//    • passBlur: 2 iterations→3 (6 passes total) for better particle merging.
//
//  Water shading model (unchanged):
//    • Schlick Fresnel  F0=0.02  (water IOR 1.33)
//    • Reflection ray transformed view→world via uViewRotInv (mat3)
//      → samples procedural zenith/horizon sky gradient (Z-up world)
//    • Beer-Lambert per-channel absorption  exp(-k·thick·(1−deepColor))
//    • Shallow/deep colour mix by accumulated thickness
//    • NdL diffuse on water body
//    • Two-lobe specular: sharp sun (pow 512) + soft sky (pow 32)
//    • Forward-scatter SSS on thin edges  exp(-k·thick)·NdL_back·shallowColor
//
//  Performance notes:
//    • All uniform locations cached at init() — zero glGetUniformLocation per frame
//    • Only 2 ping-pong FBOs (A and B) — down from 3
//    • Single blur program shared by all 6 blur passes
//    • 3 blur iterations / radius 25 (6 passes × 51 taps each)
//    • Thickness pass draws all particles additive with no depth test
//
//  settings.h additions required (add to float block):
//    float skyZenithR=0.05f, skyZenithG=0.15f, skyZenithB=0.45f;
//    float skyHorizonR=0.55f, skyHorizonG=0.75f, skyHorizonB=0.90f;
//    float reflStrength=0.70f;
//  Change shaderType default to 0.
//
//  main.cpp glClearColor should use settings.bgColorR/G/B so the BG control works:
//    glClearColor(settings.bgColorR, settings.bgColorG, settings.bgColorB, 1.0f);
// ═══════════════════════════════════════════════════════════════════════════════

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include "settings.h"
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
//  PASS 1 — DEPTH  (shared vertex shader for depth + thickness passes)
//
//  Expands billboard quads from the existing VAO:
//    attrib 0  inCenter   vec3   world-space sphere centre
//    attrib 1  inRadius   float  radius
//    attrib 2  inColor    vec4   unused but bound (stride must match)
//    attrib 3  inOffset   vec2   billboard quad offset in [-1,1]²
//
//  Fragment: reconstructs true sphere surface point in view space,
//  writes positive linear depth to R32F and overrides gl_FragDepth so
//  particles self-occlude correctly in the depth buffer.
// ─────────────────────────────────────────────────────────────────────────────
static const char* SSF_DEPTH_VERT = R"glsl(
#version 330 core
layout(location=0) in vec3  inCenter;
layout(location=1) in float inRadius;
layout(location=2) in vec4  inColor;
layout(location=3) in vec2  inOffset;
uniform mat4 uProj;
uniform mat4 uView;
out vec2  vOffset;
out float vRadius;
out vec3  vCenterView;
void main(){
    vec3 right = vec3(uView[0][0], uView[1][0], uView[2][0]);
    vec3 up    = vec3(uView[0][1], uView[1][1], uView[2][1]);
    vec3 wPos  = inCenter + (right*inOffset.x + up*inOffset.y)*inRadius;
    gl_Position = uProj * uView * vec4(wPos,1.0);
    vOffset     = inOffset;
    vRadius     = inRadius;
    vCenterView = (uView * vec4(inCenter,1.0)).xyz;
}
)glsl";

static const char* SSF_DEPTH_FRAG = R"glsl(
#version 330 core
in vec2  vOffset;
in float vRadius;
in vec3  vCenterView;
uniform mat4 uProj;
out float outLinearDepth;
void main(){
    float r2 = dot(vOffset,vOffset);
    if(r2 > 1.0) discard;
    float zLocal   = sqrt(max(1.0-r2,0.0));
    vec3  surfView = vCenterView + vec3(vOffset.x,vOffset.y,zLocal)*vRadius;
    outLinearDepth = -surfView.z;                          // positive depth
    vec4 clip    = uProj * vec4(surfView,1.0);
    gl_FragDepth = (clip.z/clip.w)*0.5 + 0.5;             // hardware depth
}
)glsl";

// ─────────────────────────────────────────────────────────────────────────────
//  PASS 2 — THICKNESS
//  Same vertex shader. Rendered additively, no depth test.
//
//  FIX v5: coefficient 0.030→0.07 (v4 used 0.12 which over-saturated thickness
//  in dense regions, causing the composite to absorb all color → black output).
//  0.07 gives meaningful thickness even at 2–3 particle overlap while staying
//  below the black-out threshold at 6–8 layers.
// ─────────────────────────────────────────────────────────────────────────────
static const char* SSF_THICK_FRAG = R"glsl(
#version 330 core
in vec2  vOffset;
in float vRadius;
out float outThickness;
void main(){
    float r2 = dot(vOffset,vOffset);
    if(r2 > 1.0) discard;
    outThickness = sqrt(max(1.0-r2,0.0)) * vRadius * 0.07;
}
)glsl";

// ─────────────────────────────────────────────────────────────────────────────
//  FULLSCREEN QUAD VERTEX  (blur + composite)
// ─────────────────────────────────────────────────────────────────────────────
static const char* SSF_QUAD_VERT = R"glsl(
#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main(){
    vUV = aPos*0.5+0.5;
    gl_Position = vec4(aPos,0.0,1.0);
}
)glsl";

// ─────────────────────────────────────────────────────────────────────────────
//  PASS 3 — SEPARABLE BILATERAL BLUR
//
//  Run 6 times total: H1→V1→H2→V2→H3→V3 (3 H+V iterations).
//  Radius 25 = 51 taps per pass.
//
//  FIX v4 — depth sigma formula:
//  Old:  wDp = exp(-|Δd| × uBlurDepthFall)
//  Problem: with uBlurDepthFall=3.0 and world-scale depths (particles at
//  depth ~80, sphere radius ~3.5), the depth difference across two adjacent
//  sphere surfaces is ~7 units → exp(-7×3) = exp(-21) ≈ 0.
//  The bilateral was rejecting every tap across particle boundaries — the
//  blur did effectively NOTHING.
//
//  New:  sigmaD = uBlurDepthFall × 2.0  (interpret as world-unit sigma)
//        wDp = exp(-Δd² / (2 × sigmaD²))
//  At uBlurDepthFall=3.0 → sigmaD=6.  Adjacent particles (Δd≈4) get
//  weight exp(-16/72)=0.80.  Across a sphere (Δd≈7): 0.51.  Particle vs
//  background (Δd≈80): ~0.  Edges still preserved, particles now merge.
//
//  Sparse-splat gap fill:
//  Empty taps (depth<0.001) inside the particle cloud are filled with the
//  center depth so the spatial Gaussian still accumulates.
// ─────────────────────────────────────────────────────────────────────────────
static const char* SSF_BLUR_FRAG = R"glsl(
#version 330 core
uniform sampler2D uDepthTex;
uniform vec2      uTexelSize;
uniform vec2      uBlurDir;
uniform float     uBlurSigma;
uniform float     uBlurDepthFall;
in  vec2  vUV;
out float outDepth;
const int RADIUS = 1;
void main(){
    float center = texture(uDepthTex,vUV).r;
    if(center < 0.001){ outDepth=0.0; return; }

    // Depth sigma in world units — makes bilateral scale-invariant.
    // uBlurDepthFall (default 3.0) → sigmaD = 6.0 world units ≈ 2× particle radius.
    float sigmaD = uBlurDepthFall * 2.0;
    float sigmaD2 = 2.0 * sigmaD * sigmaD;

    float sig2 = uBlurSigma * uBlurSigma;
    float sum  = 0.0;
    float wsum = 0.0;

    for(int i = -RADIUS; i <= RADIUS; ++i){
        vec2  off = uBlurDir * float(i) * uTexelSize;
        float raw = texture(uDepthTex, vUV + off).r;
        // Fill intra-fluid holes with center depth (sparse splat gap bridging)
        float d = (raw < 0.001) ? center : raw;

        float wSp = exp(-float(i*i) / (2.0*sig2));
        float dif = d - center;
        float wDp = exp(-(dif*dif) / sigmaD2);   // Gaussian, not linear
        float w   = wSp * wDp;
        sum  += d * w;
        wsum += w;
    }
    outDepth = (wsum > 1e-6) ? sum/wsum : center;
}
)glsl";

// ─────────────────────────────────────────────────────────────────────────────
//  PASS 4 — WATER COMPOSITE
//
//  Normal reconstruction
//    2-texel central finite difference on blurred depth.
//    Cross product of two screen-space tangents gives the outward normal.
//    In OpenGL view space (+Z toward camera):
//      cross(ddx,ddy) points +Z for a flat surface → correct outward normal.
//    Guard: if N.z < 0 the winding is reversed, flip.
//
//  Sky reflection
//    Reflection vector R = reflect(incident, N) in view space.
//    uViewRotInv = transpose(mat3(view)) transforms R to world space.
//    World is Z-up: R_world.z ∈ [0,1] → horizon-to-zenith sky gradient.
//
//  Lighting
//    fresnel     Schlick F0=0.02 (water IOR 1.33)
//    specSharp   Sun point highlight  pow(NdH, 512)
//    specSky     Soft sky reflection  pow(NdH, 32) × 0.06
//    absorbed    Beer-Lambert per channel: exp(-k·thick·(1−deepColor))
//    bodyColor   mix(deepColor, shallowColor, thinness) × absorbed
//    diffuse     bodyColor × NdL × (1−fresnel)
//    sss         Forward-scatter: thin edges glow with shallowColor when lit
//
//  FIX v4 — alpha formula:
//  Old: ad-hoc blend of depthFactor + thick + fresnel components, clamped
//       to [0.20, 0.90].  Gave ~0.40 everywhere regardless of actual fluid depth.
//  New: Beer-Lambert coverage   alpha_body = 1 − exp(−uAbsorption × thick × 0.22)
//       Fresnel adds a base reflective opacity even at zero thickness (grazing).
//       Result: edges transparent, bulk opaque, physically motivated.
//       thick clamped to 15.0 (was 3.0) to match the raised thickness coefficient.
// ─────────────────────────────────────────────────────────────────────────────
static const char* SSF_COMPOSITE_FRAG = R"glsl(
#version 330 core
uniform sampler2D uDepthTex;
uniform sampler2D uThickTex;
uniform vec2      uTexelSize;
uniform vec3      uLightDirView;    // normalised, view space
uniform mat3      uViewRotInv;      // transpose(mat3(view)) — view→world

uniform vec3  uShallowColor;        // thin/surface colour (live from settings)
uniform vec3  uDeepColor;           // thick/interior colour
uniform float uAbsorption;          // depth→color transition rate

uniform vec3  uSkyZenith;           // sky colour at zenith
uniform vec3  uSkyHorizon;          // sky colour at horizon
uniform float uReflStrength;        // reflection multiplier

uniform float uTanHalfFov;
uniform float uAspect;

in  vec2  vUV;
out vec4  outColor;

vec3 toView(vec2 uv, float d){
    vec2 ndc = uv*2.0-1.0;
    return vec3(ndc*vec2(uAspect,1.0)*uTanHalfFov*d, -d);
}

vec3 reconstructNormal(vec2 uv, float d0){
    vec2 ox = vec2(uTexelSize.x*2.0, 0.0);
    vec2 oy = vec2(0.0, uTexelSize.y*2.0);
    float dR=texture(uDepthTex,uv+ox).r, dL=texture(uDepthTex,uv-ox).r;
    float dU=texture(uDepthTex,uv+oy).r, dD=texture(uDepthTex,uv-oy).r;
    vec3 posC = toView(uv,d0);
    vec3 ddx = (dR>0.001 && dL>0.001)
        ? toView(uv+ox,dR)-toView(uv-ox,dL)
        : (dR>0.001 ? toView(uv+ox,dR)-posC : posC-toView(uv-ox,dL));
    vec3 ddy = (dU>0.001 && dD>0.001)
        ? toView(uv+oy,dU)-toView(uv-oy,dD)
        : (dU>0.001 ? toView(uv+oy,dU)-posC : posC-toView(uv-oy,dD));
    vec3 N = normalize(cross(ddx,ddy));
    if(N.z < 0.0) N = -N;
    return N;
}

vec3 sampleSky(vec3 Rw){
    float t = clamp(Rw.z, 0.0, 1.0);
    return mix(uSkyHorizon, uSkyZenith, t*t) * uReflStrength;
}

void main(){
    float depth = texture(uDepthTex,vUV).r;
    if(depth < 0.001) discard;

    float thick = clamp(texture(uThickTex,vUV).r, 0.0, 8.0);
    vec3  N = reconstructNormal(vUV, depth);
    vec3  V = vec3(0.0,0.0,1.0);

    // ── Schlick Fresnel  F0=0.02 ──────────────────────────────────────────────
    float cosV    = max(dot(N,V), 0.0);
    float fresnel = 0.02 + 0.98*pow(1.0-cosV, 5.0);

    // ── Sky reflection ────────────────────────────────────────────────────────
    vec3 R_world = uViewRotInv * reflect(-V, N);
    vec3 skyCol  = sampleSky(R_world);

    // ── Sky refraction (approximate) ──────────────────────────────────────────
    vec3 T_view = refract(-V, N, 1.0/1.333);
    vec3 T_world = uViewRotInv * T_view;
    vec3 refrSkyCol = (dot(T_view, T_view) > 0.0) ? sampleSky(T_world) : uSkyHorizon;

    // ── Sun specular ──────────────────────────────────────────────────────────
    vec3  H         = normalize(V + uLightDirView);
    float NdH       = max(dot(N,H), 0.0);
    float specSharp = pow(NdH, 512.0);
    float specSky   = pow(NdH,  32.0) * 0.06;
    vec3  specCol   = (vec3(1.0)*specSharp + uSkyZenith*specSky) * fresnel;

    // ── Water color by depth gradient (surface -> deep) ──────────────────────
    // surface: shallowColor, deep: deepColor.
    float thinFactor = exp(-uAbsorption * thick * 0.08);  // 1 at surface, 0 at depth.
    float depthLight = clamp(thick / 4.0, 0.0, 1.0);
    float colorBlend = 1.0 - exp(-uAbsorption * thick * 0.35);
    vec3  waterColor = mix(uShallowColor, uDeepColor, colorBlend);
    
    // extra color darkening in deep zones to avoid full transparency illusion
    vec3  darkening = mix(vec3(1.0), uDeepColor * 1.5, depthLight);
    waterColor *= darkening;

    // ── Diffuse lighting on water body ────────────────────────────────────────
    float NdL   = max(dot(N, uLightDirView), 0.0);
    vec3  lit   = waterColor * (0.15 + 0.85 * NdL * (1.0 - fresnel));

    // ── Forward-scatter SSS: thin edges lit from behind glow with shallowColor ─
    float sssWt  = thinFactor * max(dot(uLightDirView, -V), 0.0);
    vec3  sssCol = uShallowColor * sssWt * 0.22;

    // ── Refraction contribution from sky + water color absorption ───────────
    vec3 refraction = refrSkyCol * (1.0 - thinFactor) * 0.75;

    // ── Fresnel blend: lit body + refraction vs sky reflection ───────────────
    vec3 col = mix(lit + sssCol + refraction, skyCol, fresnel) + specCol;

    // ── Alpha: thickness-driven + grazing Fresnel ─────────────────────────────
    // Depth-based alpha mapping to keep deep areas opaque while thin edges stay clear.
    float alphaBody    = clamp(1.0 - exp(-thick * 3.0), 0.35, 0.95);
    float alphaFresnel = fresnel * 0.25;
    float alpha        = clamp(alphaBody + alphaFresnel, 0.0, 1.0);

    // (No gl_FragDepth now; preserve stable blending from legacy pipeline)
    outColor = vec4(col, alpha);
}
)glsl";

// ═══════════════════════════════════════════════════════════════════════════════
//  Cached uniform locations — populated once in init(), used every frame.
//  This eliminates glGetUniformLocation from the render hot path entirely.
// ═══════════════════════════════════════════════════════════════════════════════
struct SSFUniforms {
    // depth pass
    GLint depth_uProj = -1, depth_uView = -1;
    // thickness pass
    GLint thick_uProj = -1, thick_uView = -1;
    // blur pass (6 calls, same program)
    GLint blur_uDepthTex = -1, blur_uTexelSize = -1, blur_uBlurDir = -1;
    GLint blur_uBlurSigma = -1, blur_uBlurDepthFall = -1;
    // composite pass
    GLint comp_uDepthTex = -1, comp_uThickTex = -1, comp_uTexelSize = -1;
    GLint comp_uLightDirView = -1, comp_uViewRotInv = -1;
    GLint comp_uShallowColor = -1, comp_uDeepColor = -1, comp_uAbsorption = -1;
    GLint comp_uSkyZenith = -1, comp_uSkyHorizon = -1, comp_uReflStrength = -1;
    GLint comp_uTanHalfFov = -1, comp_uAspect = -1;
};

// ═══════════════════════════════════════════════════════════════════════════════
//  FluidRenderer
// ═══════════════════════════════════════════════════════════════════════════════
struct FluidRenderer {
    // FBOs
    GLuint depthFBO = 0;   // Pass 1: depth colour + hardware depth RBO
    GLuint blurFBO_A = 0;   // Blur ping
    GLuint blurFBO_B = 0;   // Blur pong — final blur result after 3 iterations
    GLuint thickFBO = 0;   // Pass 2: additive thickness

    // Textures
    GLuint depthTex = 0;   GLuint blurTexA = 0;
    GLuint blurTexB = 0;   GLuint thickTex = 0;
    GLuint depthRBO = 0;   // hardware depth renderbuffer for Pass 1

    // Programs  (single blur program shared by all 6 blur passes)
    GLuint depthProg = 0;
    GLuint thickProg = 0;
    GLuint blurProg = 0;
    GLuint compositeProg = 0;

    GLuint quadVAO = 0, quadVBO = 0;
    SSFUniforms u;              // cached locations

    int  width = 0, height = 0;
    bool ready = false;

    // ── Public API ─────────────────────────────────────────────────────────────
    void init(int w, int h);

    // Returns true  → SSF rendered  (shaderType 0).
    // Returns false → legacy mode   (shaderType 1), caller draws particles.
    bool render(GLuint particleVAO, int count,
        const glm::mat4& proj, const glm::mat4& view,
        int shaderType,
        const glm::vec3& lightDirWorld,
        float fovDegrees, float aspect);

    void resize(int w, int h);
    void cleanup();

private:
    GLuint makeR32FFBO(int w, int h, GLuint& texOut);
    GLuint compileShader(GLenum type, const char* src);
    GLuint linkProgram(const char* vs, const char* fs);
    void   initQuad();
    void   cacheUniforms();
    void   destroyBuffers();

    void passDepth(GLuint vao, int n, const glm::mat4& proj, const glm::mat4& view);
    void passThickness(GLuint vao, int n, const glm::mat4& proj, const glm::mat4& view);
    void runBlurPass(GLuint srcTex, GLuint dstFBO, float dx, float dy);
    void passBlur();
    void passComposite(const glm::vec3& lightDirView,
        const glm::mat3& viewRotInv,
        float fovDeg, float aspect);
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Implementation
// ═══════════════════════════════════════════════════════════════════════════════

inline GLuint FluidRenderer::compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr); glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char b[2048]; glGetShaderInfoLog(s, 2048, nullptr, b);
        std::cerr << "[FluidRenderer] compile:\n" << b << "\n";
    }
    return s;
}

inline GLuint FluidRenderer::linkProgram(const char* vs, const char* fs) {
    GLuint a = compileShader(GL_VERTEX_SHADER, vs);
    GLuint b = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, a); glAttachShader(p, b); glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[2048]; glGetProgramInfoLog(p, 2048, nullptr, buf);
        std::cerr << "[FluidRenderer] link:\n" << buf << "\n";
    }
    glDeleteShader(a); glDeleteShader(b);
    return p;
}

inline GLuint FluidRenderer::makeR32FFBO(int w, int h, GLuint& texOut) {
    GLuint fbo;
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &texOut);  glBindTexture(GL_TEXTURE_2D, texOut);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texOut, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return fbo;
}

inline void FluidRenderer::initQuad() {
    float v[] = { -1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1 };
    glGenVertexArrays(1, &quadVAO); glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);
}

// Cache every uniform location once after programs are compiled
inline void FluidRenderer::cacheUniforms() {
    // depth
    u.depth_uProj = glGetUniformLocation(depthProg, "uProj");
    u.depth_uView = glGetUniformLocation(depthProg, "uView");
    // thickness
    u.thick_uProj = glGetUniformLocation(thickProg, "uProj");
    u.thick_uView = glGetUniformLocation(thickProg, "uView");
    // blur
    u.blur_uDepthTex = glGetUniformLocation(blurProg, "uDepthTex");
    u.blur_uTexelSize = glGetUniformLocation(blurProg, "uTexelSize");
    u.blur_uBlurDir = glGetUniformLocation(blurProg, "uBlurDir");
    u.blur_uBlurSigma = glGetUniformLocation(blurProg, "uBlurSigma");
    u.blur_uBlurDepthFall = glGetUniformLocation(blurProg, "uBlurDepthFall");
    // composite
    u.comp_uDepthTex = glGetUniformLocation(compositeProg, "uDepthTex");
    u.comp_uThickTex = glGetUniformLocation(compositeProg, "uThickTex");
    u.comp_uTexelSize = glGetUniformLocation(compositeProg, "uTexelSize");
    u.comp_uLightDirView = glGetUniformLocation(compositeProg, "uLightDirView");
    u.comp_uViewRotInv = glGetUniformLocation(compositeProg, "uViewRotInv");
    u.comp_uShallowColor = glGetUniformLocation(compositeProg, "uShallowColor");
    u.comp_uDeepColor = glGetUniformLocation(compositeProg, "uDeepColor");
    u.comp_uAbsorption = glGetUniformLocation(compositeProg, "uAbsorption");
    u.comp_uSkyZenith = glGetUniformLocation(compositeProg, "uSkyZenith");
    u.comp_uSkyHorizon = glGetUniformLocation(compositeProg, "uSkyHorizon");
    u.comp_uReflStrength = glGetUniformLocation(compositeProg, "uReflStrength");
    u.comp_uTanHalfFov = glGetUniformLocation(compositeProg, "uTanHalfFov");
    u.comp_uAspect = glGetUniformLocation(compositeProg, "uAspect");
}

inline void FluidRenderer::destroyBuffers() {
    GLuint fbos[] = { depthFBO,blurFBO_A,blurFBO_B,thickFBO };
    GLuint texs[] = { depthTex,blurTexA, blurTexB, thickTex };
    glDeleteFramebuffers(4, fbos); glDeleteTextures(4, texs);
    glDeleteRenderbuffers(1, &depthRBO);
    depthFBO = blurFBO_A = blurFBO_B = thickFBO = 0;
    depthTex = blurTexA = blurTexB = thickTex = depthRBO = 0;
}

inline void FluidRenderer::init(int w, int h) {
    width = w; height = h;

    depthProg = linkProgram(SSF_DEPTH_VERT, SSF_DEPTH_FRAG);
    thickProg = linkProgram(SSF_DEPTH_VERT, SSF_THICK_FRAG);
    blurProg = linkProgram(SSF_QUAD_VERT, SSF_BLUR_FRAG);
    compositeProg = linkProgram(SSF_QUAD_VERT, SSF_COMPOSITE_FRAG);
    cacheUniforms();

    // Depth FBO: R32F colour + hardware depth renderbuffer
    glGenFramebuffers(1, &depthFBO); glBindFramebuffer(GL_FRAMEBUFFER, depthFBO);
    glGenTextures(1, &depthTex);     glBindTexture(GL_TEXTURE_2D, depthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depthTex, 0);
    glGenRenderbuffers(1, &depthRBO); glBindRenderbuffer(GL_RENDERBUFFER, depthRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRBO);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    blurFBO_A = makeR32FFBO(w, h, blurTexA);
    blurFBO_B = makeR32FFBO(w, h, blurTexB);
    thickFBO = makeR32FFBO(w, h, thickTex);
    initQuad();
    ready = true;
}

inline void FluidRenderer::resize(int w, int h) {
    destroyBuffers(); width = w; height = h;
    glGenFramebuffers(1, &depthFBO); glBindFramebuffer(GL_FRAMEBUFFER, depthFBO);
    glGenTextures(1, &depthTex);     glBindTexture(GL_TEXTURE_2D, depthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depthTex, 0);
    glGenRenderbuffers(1, &depthRBO); glBindRenderbuffer(GL_RENDERBUFFER, depthRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRBO);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    blurFBO_A = makeR32FFBO(w, h, blurTexA);
    blurFBO_B = makeR32FFBO(w, h, blurTexB);
    thickFBO = makeR32FFBO(w, h, thickTex);
}

// ─────────────────────────────────────────────────────────────────────────────
inline void FluidRenderer::passDepth(GLuint vao, int n,
    const glm::mat4& proj, const glm::mat4& view) {
    glBindFramebuffer(GL_FRAMEBUFFER, depthFBO);
    glViewport(0, 0, width, height);
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS); glDisable(GL_BLEND);
    glUseProgram(depthProg);
    glUniformMatrix4fv(u.depth_uProj, 1, GL_FALSE, glm::value_ptr(proj));
    glUniformMatrix4fv(u.depth_uView, 1, GL_FALSE, glm::value_ptr(view));
    glBindVertexArray(vao); glDrawArrays(GL_TRIANGLES, 0, n * 3); glBindVertexArray(0);
    glUseProgram(0); glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

inline void FluidRenderer::passThickness(GLuint vao, int n,
    const glm::mat4& proj, const glm::mat4& view) {
    glBindFramebuffer(GL_FRAMEBUFFER, thickFBO);
    glViewport(0, 0, width, height);
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_ONE, GL_ONE);
    glUseProgram(thickProg);
    glUniformMatrix4fv(u.thick_uProj, 1, GL_FALSE, glm::value_ptr(proj));
    glUniformMatrix4fv(u.thick_uView, 1, GL_FALSE, glm::value_ptr(view));
    glBindVertexArray(vao); glDrawArrays(GL_TRIANGLES, 0, n * 3); glBindVertexArray(0);
    glUseProgram(0); glDisable(GL_BLEND); glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// Single separable blur pass — src → dst
inline void FluidRenderer::runBlurPass(GLuint srcTex, GLuint dstFBO, float dx, float dy) {
    glBindFramebuffer(GL_FRAMEBUFFER, dstFBO);
    glViewport(0, 0, width, height); glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND);
    glUseProgram(blurProg);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, srcTex);
    glUniform1i(u.blur_uDepthTex, 0);
    glUniform2f(u.blur_uTexelSize, 1.0f / width, 1.0f / height);
    glUniform2f(u.blur_uBlurDir, dx, dy);
    glUniform1f(u.blur_uBlurSigma, settings.blurSigma);
    glUniform1f(u.blur_uBlurDepthFall, settings.blurDepthFall);
    glBindVertexArray(quadVAO); glDrawArrays(GL_TRIANGLES, 0, 6); glBindVertexArray(0);
    glUseProgram(0); glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// 3 H+V iterations (6 passes).  Final result lands in blurTexB.
//   it1: depthTex →H→ blurTexA →V→ blurTexB
//   it2: blurTexB →H→ blurTexA →V→ blurTexB
//   it3: blurTexB →H→ blurTexA →V→ blurTexB
//
// FIX v4: added 3rd iteration so sparse inter-particle gaps fully close
// when the camera is zoomed in close.
inline void FluidRenderer::passBlur() {
    runBlurPass(depthTex, blurFBO_A, 1, 0);   // it1 H
    runBlurPass(blurTexA, blurFBO_B, 0, 1);   // it1 V
    runBlurPass(blurTexB, blurFBO_A, 1, 0);   // it2 H
    runBlurPass(blurTexA, blurFBO_B, 0, 1);   // it2 V
    runBlurPass(blurTexB, blurFBO_A, 1, 0);   // it3 H
    runBlurPass(blurTexA, blurFBO_B, 0, 1);   // it3 V  ← final in blurTexB
}

inline void FluidRenderer::passComposite(const glm::vec3& lightDirView,
    const glm::mat3& viewRotInv,
    float fovDeg, float aspect) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, width, height);
    glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    float tanHalf = tanf(glm::radians(fovDeg * 0.5f));

    glUseProgram(compositeProg);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, blurTexB);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, thickTex);

    glUniform1i(u.comp_uDepthTex, 0);
    glUniform1i(u.comp_uThickTex, 1);
    glUniform2f(u.comp_uTexelSize, 1.0f / width, 1.0f / height);
    glUniform3fv(u.comp_uLightDirView, 1, glm::value_ptr(lightDirView));
    glUniformMatrix3fv(u.comp_uViewRotInv, 1, GL_FALSE, glm::value_ptr(viewRotInv));

    // Water body colours — read live from settings every frame
    glUniform3f(u.comp_uShallowColor,
        settings.shallowColorR, settings.shallowColorG, settings.shallowColorB);
    glUniform3f(u.comp_uDeepColor,
        settings.deepColorR, settings.deepColorG, settings.deepColorB);
    glUniform1f(u.comp_uAbsorption, settings.absorption);

    // Sky colours — read live from settings every frame
    glUniform3f(u.comp_uSkyZenith,
        settings.skyZenithR, settings.skyZenithG, settings.skyZenithB);
    glUniform3f(u.comp_uSkyHorizon,
        settings.skyHorizonR, settings.skyHorizonG, settings.skyHorizonB);
    glUniform1f(u.comp_uReflStrength, settings.reflStrength);

    glUniform1f(u.comp_uTanHalfFov, tanHalf);
    glUniform1f(u.comp_uAspect, aspect);

    glBindVertexArray(quadVAO); glDrawArrays(GL_TRIANGLES, 0, 6); glBindVertexArray(0);
    glUseProgram(0); glDisable(GL_BLEND);
}

inline bool FluidRenderer::render(GLuint particleVAO, int particleCount,
    const glm::mat4& proj, const glm::mat4& view,
    int shaderType,
    const glm::vec3& lightDirWorld,
    float fovDegrees, float aspect) {
    // ── shaderType 1: legacy particles — COMPLETELY UNCHANGED ─────────────────
    if (shaderType == 1) {
        settings.heateffect = true;
        return false;
    }
    if (!ready || particleCount <= 0) return false;

    // Heat colouring is invisible under the continuous SSF surface
    settings.heateffect = false;

    // Light direction in view space
    glm::vec3 lightDirView = glm::normalize(
        glm::vec3(view * glm::vec4(lightDirWorld, 0.0f)));

    // View rotation inverse: transforms reflected ray from view space to world space.
    // = transpose(mat3(view)) because view rotation is orthonormal.
    glm::mat3 viewRotInv = glm::transpose(glm::mat3(view));

    passDepth(particleVAO, particleCount, proj, view);
    passThickness(particleVAO, particleCount, proj, view);
    passBlur();
    passComposite(lightDirView, viewRotInv, fovDegrees, aspect);
    return true;
}

inline void FluidRenderer::cleanup() {
    destroyBuffers();
    if (quadVAO) { glDeleteVertexArrays(1, &quadVAO); quadVAO = 0; }
    if (quadVBO) { glDeleteBuffers(1, &quadVBO);       quadVBO = 0; }
    GLuint progs[] = { depthProg,thickProg,blurProg,compositeProg };
    for (GLuint p : progs) if (p) glDeleteProgram(p);
    depthProg = thickProg = blurProg = compositeProg = 0;
    ready = false;
}