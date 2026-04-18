#pragma once
// ═══════════════════════════════════════════════════════════════════════════════
//  fluid_renderer.h  v4
//
//  shaderType 0 → Screen-space water    (this renderer — 5 passes)
//  shaderType 1 → Legacy particles      (render() returns false, caller draws)
//
//  Pipeline (mode 0):
//    Pass 1  Depth        sphere-surface depth → R32F FBO + gl_FragDepth
//    Pass 2  Thickness    flat 0.1 per hit pixel (additive, Beer-Lambert input)
//    Pass 3  Pack         merge depthTex + thickTex → packTex (RGBA32F)
//    Pass 4  Blur ×4      separable adaptive-radius bilateral on RGBA32F
//    Pass 5  Composite    full Fresnel, Snell refraction, Beer-Lambert
//
//  Changes from v3:
//    • SSF_THICK_FRAG     flat 0.1 contribution (Unity reference)
//    • SSF_PACK_FRAG      new pass: packs depth+thick into RGBA32F
//    • SSF_BLUR_FRAG      adaptive world-radius bilateral (replaces fixed sigma)
//    • SSF_COMPOSITE_FRAG full Fresnel equations + Snell refraction + proj exit
//    • blurFBO_A/B        upgraded to RGBA32F
//    • packFBO/packTex    new RGBA32F FBO
//    • packProg           new shader program
//    • SSFUniforms        updated (sky/viewRotInv/shadow/refrStrength removed;
//                         pack/blur/comp new uniforms added)
//    • FluidRenderer      struct updated; passComposite signature gains proj
//    • render()           passes proj to passComposite, drops viewRotInv
//
//  UNCHANGED:
//    SSF_DEPTH_VERT, SSF_DEPTH_FRAG, SSF_QUAD_VERT
//    passDepth(), passThickness(), compileShader(), linkProgram(), initQuad()
//    render() mode-1 early return block
// ═══════════════════════════════════════════════════════════════════════════════

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include "settings.h"
#include <cmath>
#include "sky.h"

// ─────────────────────────────────────────────────────────────────────────────
//  PASS 1 — DEPTH  (shared vertex shader for depth + thickness passes)
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
    gl_FragDepth = (clip.z/clip.w)*0.7 + 0.3;             // hardware depth
}
)glsl";

// ─────────────────────────────────────────────────────────────────────────────
//  PASS 2 — THICKNESS
//  Flat contribution of 0.1 per hit pixel (Unity reference behaviour).
// ─────────────────────────────────────────────────────────────────────────────
static const char* SSF_THICK_FRAG = R"glsl(
#version 330 core
in vec2  vOffset;
in float vRadius;
out float outThickness;
void main(){
    float r2 = dot(vOffset,vOffset);
    if(r2 > 1.0) {discard;}
  //  outThickness = 0.1;
 outThickness = 2.0 * sqrt(max(1.0 - r2, 0.0)) * vRadius;
}
)glsl";

// ─────────────────────────────────────────────────────────────────────────────
//  FULLSCREEN QUAD VERTEX  (pack + blur + composite)
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
//  PASS 3 — PACK
//
//  Packs separate R32F depth and R32F thickness textures into a single RGBA32F
//  texture before bilateral blur:
//    .r = smooth depth  (updated each blur iteration)
//    .g = smooth thick  (updated each blur iteration)
//    .b = unused
//    .a = HARD depth    (NEVER changes — bilateral depth-weight reference)
//
//  Matches Unity SmoothThickPrepare.shader exactly.
// ─────────────────────────────────────────────────────────────────────────────
static const char* SSF_PACK_FRAG = R"glsl(
#version 330 core
uniform sampler2D uDepthTex;
uniform sampler2D uThickTex;
in  vec2 vUV;
out vec4 outPacked1;
void main(){
    float depth = texture(uDepthTex, vUV).r;
    float thick = texture(uThickTex, vUV).r;
    outPacked1 = vec4(depth, thick, 0.0, depth);
}
)glsl";

// ─────────────────────────────────────────────────────────────────────────────
//  PASS 4 — ADAPTIVE BILATERAL BLUR
//
//  Port of Unity BilateralPass.hlsl + BilateralFilter1D.shader.
//  Operates on RGBA32F packed texture:
//    .a = hard depth — never modified, used as bilateral weight reference
//    .r = smooth depth — filtered each pass
//    .g = smooth thick — filtered each pass using same weights
//    .b = unused
//
//  Kernel radius scales with world-space particle radius projected to screen,
//  so near particles receive a wider kernel (more merging) and far ones a
//  narrower one — unlike the old fixed-radius approach.
//
//  Uniforms:
//    uBlurWorldRadius   world-space kernel radius
//    uBlurStrength      sigma scale  (sigma = radius / (6 * strength))
//    uBlurDiffStrength  depth-similarity falloff exponent
//    uBlurMaxRadius     screen-space pixel cap
//    uProjScale         (screenWidth * P[0][0]) / 2  — used to project radius
//    uTexelSize         1/width, 1/height
//    uBlurDir           (1,0) or (0,1)
// ─────────────────────────────────────────────────────────────────────────────
static const char* SSF_BLUR_FRAG = R"glsl(
#version 330 core
uniform sampler2D uPackTex;
uniform vec2      uTexelSize;
uniform vec2      uBlurDir;
uniform float     uBlurWorldRadius;
uniform float     uBlurStrength;
uniform float     uBlurDiffStrength;
 uniform float uBlurParticleRadius;
uniform int       uBlurMaxRadius;
uniform float     uProjScale;
in  vec2 vUV;
out vec4 outPacked;

void main(){
    vec4 center = texture(uPackTex, vUV);
    float depth = center.a;   // hard depth -- never changes

    // Background pixel -- skip blur, pass through unchanged
    if(depth >= 10000.0){
        outPacked = center;
        return;
    }
    if(depth < 0.001){    outPacked = center; return; }

    // Compute screen-space kernel radius from world-space radius at this depth
    float pxPerUnit  = uProjScale / depth;
    float radiusFloat = pxPerUnit * uBlurWorldRadius;
    int   radius      = int(ceil(radiusFloat));
    if(radius <= 1 && uBlurWorldRadius > 0.0) radius = 2;
    if(radius > uBlurMaxRadius) radius = uBlurMaxRadius;

    float fR    = max(0.0, float(radius) - radiusFloat);   // fractional leftover
    float sigma = max(1e-7, (float(radius) - fR) / (6.0 * max(0.001, uBlurStrength)));

    // smoothMask: smooth .r (depth) and .g (thick), leave .b alone
    const vec3 smoothMask = vec3(1.0, 1.0, 0.0);

    vec4  sum  = vec4(0.0);
    float wSum = 0.0;
    vec2  texelDelta = uTexelSize * uBlurDir;

    for(int x = -radius; x <= radius; x++){
        vec4  s = texture(uPackTex, vUV + texelDelta * float(x));

        // Depth difference uses hard depth (.a) -- the reference that never blurs
        float centreDiff = depth - s.a;
  float normDiff   = centreDiff / (2.0 * uBlurParticleRadius + 0.001);
  float bgGuard    = step(0.001, s.a);
  float diffWeight = bgGuard * exp(-normDiff * normDiff * uBlurDiffStrength);
  float w          = exp(-float(x*x) / (2.0*sigma*sigma)) * diffWeight;

        sum  += s * w;
        wSum += w;
    }

    if(wSum > 0.0) sum /= wSum;

    // Apply smooth mask: lerp smoothed .rgb back; .a is always the original hard depth
    vec3 blendedRGB = mix(center.rgb, sum.rgb, smoothMask);
    outPacked = vec4(blendedRGB, depth);  // hard depth preserved in alpha
}
)glsl";

// ─────────────────────────────────────────────────────────────────────────────
//  PASS 5 — COMPOSITE
//
//  Reads blurTexB (RGBA32F packed):
//    .r = smooth depth  (for toView + normal reconstruction)
//    .g = smooth thick  (for Beer-Lambert)
//    .a = hard depth    (for early-out guard)
//
//  Lighting model (simplified, no sky reflection):
//    Full Fresnel equations (IOR air=1.0, water=1.33)
//    Full Snell's law refraction + view-space ray march + screen projection
//    Beer-Lambert per-channel extinction
//    Basic NdL diffuse + ambient (lighting placeholder)
//    NO sky reflection, NO specular, NO SSS, NO fake shadow
//
//  Normal reconstruction: Unity NormalsFromDepth approach — single-texel
//  offsets, picks the finite difference with smaller Z delta per axis.
// ─────────────────────────────────────────────────────────────────────────────
static const char* SSF_COMPOSITE_FRAG = R"glsl(
#version 330 core
uniform sampler2D uPackTex;      // RGBA32F: (smoothDepth, smoothThick, 0, hardDepth)
uniform sampler2D uSceneTex;

uniform vec2  uTexelSize;
uniform vec3  uLightDirView;     // normalised, view space
uniform float uAbsorption;
uniform vec3  uExtinction;
uniform float uTanHalfFov;
uniform float uAspect;
uniform mat4  uProj;             // projection matrix (for refraction exit projection)
uniform float uRefrMult; 
        // refraction ray march multiplier
  uniform vec3 uWaterShallow;   // shallow water tint (RGB 0-1)
  uniform vec3 uWaterDeep;      // deep water tint (RGB 0-1)

in  vec2 vUV;
out vec4 outColor;

// Unproject UV + positive linear depth to view-space position
vec3 toView(vec2 uv, float d){
  vec2 ndc = uv * 2.0 - 1.0;
   return vec3(ndc*vec2(uAspect,1.0)*uTanHalfFov*d, -d);
}

// Full Fresnel reflectance (Fresnel equations), IOR water=1.33
float calcFresnel(vec3 incident, vec3 N, float iorA, float iorB){
    float r     = iorA / iorB;
    float cosI  = clamp(-dot(incident, N), 0.0, 1.0);
    float sinSqT = r * r * (1.0 - cosI * cosI);
    if(sinSqT >= 1.0) return 1.0;   // total internal reflection
    float cosT  = sqrt(1.0 - sinSqT);
    float rPerp = (iorA * cosI - iorB * cosT) / (iorA * cosI + iorB * cosT);
    float rPar  = (iorB * cosI - iorA * cosT) / (iorB * cosI + iorA * cosT);
    return (rPerp * rPerp + rPar * rPar) * 0.5;
}

// Full Snell's law refraction direction in view space.
// Returns vec3(0) on total internal reflection.
vec3 calcRefract(vec3 incident, vec3 N, float iorA, float iorB){
    float r = iorA / iorB;
    float cosI = clamp(-dot(incident, N), 0.0, 1.0);
    float sinSqT = r * r * (1.0 - cosI * cosI);

    if (sinSqT > 1.0) {
        return vec3(0.0);
    }

    float cosT = sqrt(1.0 - sinSqT);
    return r * incident + (r * cosI - cosT) * N;
}

// Normal reconstruction -- Unity NormalsFromDepth approach.
// Single-texel offsets. Picks smaller-Z-delta finite difference per axis.
// Hard depth (.a) used for neighbour validity; smooth depth (.r) for positions.
vec3 reconstructNormal(vec2 uv, float filtD) {
    vec2 ox = vec2(uTexelSize.x, 0.0);
    vec2 oy = vec2(0.0, uTexelSize.y);
    float d00 = texture(uPackTex, uv - ox - oy).r;
    float d10 = texture(uPackTex, uv      - oy).r;
    float d20 = texture(uPackTex, uv + ox - oy).r;
    float d01 = texture(uPackTex, uv - ox     ).r;
    float d21 = texture(uPackTex, uv + ox     ).r;
    float d02 = texture(uPackTex, uv - ox + oy).r;
    float d12 = texture(uPackTex, uv      + oy).r;
    float d22 = texture(uPackTex, uv + ox + oy).r;
    float h00 = step(0.001, texture(uPackTex, uv - ox - oy).a);
    float h20 = step(0.001, texture(uPackTex, uv + ox - oy).a);
    float h02 = step(0.001, texture(uPackTex, uv - ox + oy).a);
    float h22 = step(0.001, texture(uPackTex, uv + ox + oy).a);
    float gx = (-d00*h00 + d20*h20) + 2.0*(-d01 + d21)
              + (-d02*h02 + d22*h22);
    float gy = (-d00*h00 - d20*h20) + 2.0*(-d10 + d12)
              + ( d02*h02 + d22*h22);
    // Correct normal from Sobel depth gradient.
    // Sobel kernel sums to 8× per-texel change; UV→view_x has 2× from NDC mapping.
    // Combined factor: 16 * texelSize.x * aspect * scale, where scale = depth * tanHalfFov.
    // Previous bug: Z was "2.0 * uTexelSize.x" (≈0.001), making every normal near-horizontal
    // → Fresnel ≈ 1.0 everywhere → full white. Fixed below.
    float scale = filtD * uTanHalfFov;
    vec3 N = normalize(vec3(-gx,
                           -gy,
                            16.0 * uTexelSize.x * uAspect * scale));
    if(N.z < 0.0) N = -N;
    return N;
}


void main(){
    vec4 packedtex    = texture(uPackTex, vUV);
    float hardDepth  = packedtex.a;
    float filtDepth = packedtex.r;
    float thick      = clamp(packedtex.g, 0.0, 8.0);

    // Early out -- use hard depth to avoid false hits from blur bleed
    if(hardDepth < 0.001){
     discard;
    }

    vec3 posV = toView(vUV, filtDepth);
    vec3 N    = reconstructNormal(vUV, filtDepth);
    vec3 V    = normalize(-posV);   // toward camera

    // -- Full Fresnel (IOR 1.0 -> 1.33) ----------------------------------------
    vec3 incident = -V;
    float fresnel = calcFresnel(incident, N, 1.0, 1.33);

    // -- Full Snell's law refraction -------------------------------------------
    vec3 refractDir = calcRefract(incident, N, 1.0, 1.33);

    vec2 refrUV;
    if(length(refractDir) > 0.001){
        // March refracted ray through fluid volume by thickness
        vec3 exitPosV  = posV + refractDir * thick * uRefrMult;
        // Project exit point back to screen UV
        vec4 clipExit  = uProj * vec4(exitPosV, 1.0);
        refrUV = clipExit.xy / clipExit.w * 0.5 + 0.5;
        refrUV = clamp(refrUV, vec2(0.001), vec2(0.999));
    } else {
        refrUV = vUV;   // TIR -- sample straight through
    }
    vec3 sceneCol = texture(uSceneTex, refrUV).rgb;

    vec3  extinction    = max(uExtinction, vec3(0.001));
  // sqrt-compress thickness to level additive-blend humps
  float thickComp    = sqrt(clamp(thick, 0.0, 8.0));
  vec3  transmittance = exp(-uAbsorption * extinction * thickComp);

  // Water tint: interpolate shallow<->deep by per-channel transmittance
  vec3 waterTint = mix(uWaterDeep, uWaterShallow, transmittance);

  // Surface lighting — independent of transmittance
  float NdL        = max(dot(N, uLightDirView), 0.0);
  vec3  surfLight  = waterTint * (NdL * (1.0 - fresnel) + 0.15);

  // Correct composite: background attenuated + fluid body fills the rest
  vec3 refracted   = sceneCol * transmittance + surfLight * (1.0 - transmittance);

  // Fresnel sky reflection — use actual sky blue, not near-white.
  // Old: mix(..., vec3(0.85,0.90,0.95), fresnel*0.6) — essentially a white mirror.
  vec3 col = mix(refracted, vec3(0.50, 0.65, 0.85), fresnel * 0.35);
  outColor = vec4(col, 1.0);

}
)glsl";

// ═══════════════════════════════════════════════════════════════════════════════
//  Cached uniform locations
// ═══════════════════════════════════════════════════════════════════════════════
struct SSFUniforms
{
    // depth pass
    GLint depth_uProj = -1, depth_uView = -1;
    // thickness pass
    GLint thick_uProj = -1, thick_uView = -1;
    // pack pass
    GLint pack_uDepthTex = -1, pack_uThickTex = -1;
    // blur pass (4 calls, same program)
    GLint blur_uPackTex = -1, blur_uTexelSize = -1, blur_uBlurDir = -1;
    GLint blur_uBlurWorldRadius = -1, blur_uBlurStrength = -1;
    GLint blur_uBlurDiffStrength = -1, blur_uBlurMaxRadius = -1, blur_uProjScale = -1;
    GLint blur_uBlurParticleRadius = -1;
    // composite pass
    GLint comp_uPackTex = -1, comp_uSceneTex = -1, comp_uTexelSize = -1;
    GLint comp_uLightDirView = -1;
    GLint comp_uAbsorption = -1;
    GLint comp_uExtinction = -1;
    GLint comp_uTanHalfFov = -1, comp_uAspect = -1;
    GLint comp_uProj = -1;
    GLint comp_uRefrMult = -1;
    GLint comp_uWaterShallow = -1, comp_uWaterDeep = -1;
};

// ═══════════════════════════════════════════════════════════════════════════════
//  FluidRenderer
// ═══════════════════════════════════════════════════════════════════════════════
struct FluidRenderer
{
    // FBOs
    GLuint depthFBO = 0;  // Pass 1: depth colour + hardware depth RBO
    GLuint thickFBO = 0;  // Pass 2: additive thickness
    GLuint packFBO = 0;   // Pass 3: packed RGBA32F (depth, thick, 0, hardDepth)
    GLuint blurFBO_A = 0; // Blur ping  (RGBA32F)
    GLuint blurFBO_B = 0; // Blur pong  (RGBA32F)

    // Textures
    GLuint depthTex = 0;
    GLuint thickTex = 0;
    GLuint packTex = 0;
    GLuint blurTexA = 0;
    GLuint blurTexB = 0;
    GLuint sceneTex = 0; // copy of the scene behind the fluid
    GLuint depthRBO = 0; // hardware depth renderbuffer for Pass 1

    // Programs
    GLuint depthProg = 0;
    GLuint thickProg = 0;
    GLuint packProg = 0;
    GLuint blurProg = 0;
    GLuint compositeProg = 0;

    GLuint quadVAO = 0, quadVBO = 0;
    SSFUniforms u;

    int width = 0, height = 0;
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
    GLuint makeRGBA32FFBO(int w, int h, GLuint& texOut);
    GLuint compileShader(GLenum type, const char* src);
    GLuint linkProgram(const char* vs, const char* fs);
    void initQuad();
    void cacheUniforms();
    void destroyBuffers();
    void captureSceneColor();

    void passDepth(GLuint vao, int n, const glm::mat4& proj, const glm::mat4& view);
    void passThickness(GLuint vao, int n, const glm::mat4& proj, const glm::mat4& view);
    void passPack();
    void runBlurPass(GLuint srcTex, GLuint dstFBO, float dx, float dy,
        float projScale);
    void passBlur(const glm::mat4& proj);
    void passComposite(const glm::vec3& lightDirView,
        float fovDeg, float aspect,
        const glm::mat4& proj);
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Implementation
// ═══════════════════════════════════════════════════════════════════════════════

inline GLuint FluidRenderer::compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char b[2048];
        glGetShaderInfoLog(s, 2048, nullptr, b);
        std::cerr << "[FluidRenderer] compile:\n"
            << b << "\n";
    }
    return s;
}

inline GLuint FluidRenderer::linkProgram(const char* vs, const char* fs)
{
    GLuint a = compileShader(GL_VERTEX_SHADER, vs);
    GLuint b = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, a);
    glAttachShader(p, b);
    glLinkProgram(p);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char buf[2048];
        glGetProgramInfoLog(p, 2048, nullptr, buf);
        std::cerr << "[FluidRenderer] link:\n"
            << buf << "\n";
    }
    glDeleteShader(a);
    glDeleteShader(b);
    return p;
}

inline GLuint FluidRenderer::makeR32FFBO(int w, int h, GLuint& texOut)
{
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &texOut);
    glBindTexture(GL_TEXTURE_2D, texOut);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texOut, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return fbo;
}

inline GLuint FluidRenderer::makeRGBA32FFBO(int w, int h, GLuint& texOut)
{
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &texOut);
    glBindTexture(GL_TEXTURE_2D, texOut);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texOut, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return fbo;
}

inline void FluidRenderer::initQuad()
{
    float v[] = { -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1 };
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);
}

inline void FluidRenderer::cacheUniforms()
{
    // depth
    u.depth_uProj = glGetUniformLocation(depthProg, "uProj");
    u.depth_uView = glGetUniformLocation(depthProg, "uView");
    // thickness
    u.thick_uProj = glGetUniformLocation(thickProg, "uProj");
    u.thick_uView = glGetUniformLocation(thickProg, "uView");
    // pack
    u.pack_uDepthTex = glGetUniformLocation(packProg, "uDepthTex");
    u.pack_uThickTex = glGetUniformLocation(packProg, "uThickTex");
    // blur
    u.blur_uPackTex = glGetUniformLocation(blurProg, "uPackTex");
    u.blur_uTexelSize = glGetUniformLocation(blurProg, "uTexelSize");
    u.blur_uBlurDir = glGetUniformLocation(blurProg, "uBlurDir");
    u.blur_uBlurWorldRadius = glGetUniformLocation(blurProg, "uBlurWorldRadius");
    u.blur_uBlurStrength = glGetUniformLocation(blurProg, "uBlurStrength");
    u.blur_uBlurDiffStrength = glGetUniformLocation(blurProg, "uBlurDiffStrength");
    u.blur_uBlurParticleRadius = glGetUniformLocation(blurProg, "uBlurParticleRadius");
    u.blur_uBlurMaxRadius = glGetUniformLocation(blurProg, "uBlurMaxRadius");
    u.blur_uProjScale = glGetUniformLocation(blurProg, "uProjScale");
    // composite
    u.comp_uPackTex = glGetUniformLocation(compositeProg, "uPackTex");
    u.comp_uSceneTex = glGetUniformLocation(compositeProg, "uSceneTex");
    u.comp_uTexelSize = glGetUniformLocation(compositeProg, "uTexelSize");
    u.comp_uLightDirView = glGetUniformLocation(compositeProg, "uLightDirView");
    u.comp_uAbsorption = glGetUniformLocation(compositeProg, "uAbsorption");
    u.comp_uExtinction = glGetUniformLocation(compositeProg, "uExtinction");
    u.comp_uTanHalfFov = glGetUniformLocation(compositeProg, "uTanHalfFov");
    u.comp_uAspect = glGetUniformLocation(compositeProg, "uAspect");
    u.comp_uProj = glGetUniformLocation(compositeProg, "uProj");
    u.comp_uRefrMult = glGetUniformLocation(compositeProg, "uRefrMult");
    u.comp_uWaterShallow = glGetUniformLocation(compositeProg, "uWaterShallow");
    u.comp_uWaterDeep = glGetUniformLocation(compositeProg, "uWaterDeep");

}

inline void FluidRenderer::destroyBuffers()
{
    GLuint fbos[] = { depthFBO, thickFBO, packFBO, blurFBO_A, blurFBO_B };
    GLuint texs[] = { depthTex, thickTex, packTex, blurTexA, blurTexB, sceneTex };
    glDeleteFramebuffers(5, fbos);
    glDeleteTextures(6, texs);
    glDeleteRenderbuffers(1, &depthRBO);
    depthFBO = thickFBO = packFBO = blurFBO_A = blurFBO_B = 0;
    depthTex = thickTex = packTex = blurTexA = blurTexB = sceneTex = depthRBO = 0;
}

inline void FluidRenderer::init(int w, int h)
{
    width = w;
    height = h;

    depthProg = linkProgram(SSF_DEPTH_VERT, SSF_DEPTH_FRAG);
    thickProg = linkProgram(SSF_DEPTH_VERT, SSF_THICK_FRAG);
    packProg = linkProgram(SSF_QUAD_VERT, SSF_PACK_FRAG);
    blurProg = linkProgram(SSF_QUAD_VERT, SSF_BLUR_FRAG);
    compositeProg = linkProgram(SSF_QUAD_VERT, SSF_COMPOSITE_FRAG);
    cacheUniforms();

    // Depth FBO: R32F colour + hardware depth renderbuffer
    glGenFramebuffers(1, &depthFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, depthFBO);
    glGenTextures(1, &depthTex);
    glBindTexture(GL_TEXTURE_2D, depthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depthTex, 0);
    glGenRenderbuffers(1, &depthRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRBO);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    thickFBO = makeR32FFBO(w, h, thickTex);
    packFBO = makeRGBA32FFBO(w, h, packTex);
    blurFBO_A = makeRGBA32FFBO(w, h, blurTexA);
    blurFBO_B = makeRGBA32FFBO(w, h, blurTexB);

    glGenTextures(1, &sceneTex);
    glBindTexture(GL_TEXTURE_2D, sceneTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    initQuad();
    ready = true;
}

inline void FluidRenderer::resize(int w, int h)
{
    if (!ready)
    {
        width = w;
        height = h;
        return;
    }
    destroyBuffers();
    width = w;
    height = h;

    glGenFramebuffers(1, &depthFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, depthFBO);
    glGenTextures(1, &depthTex);
    glBindTexture(GL_TEXTURE_2D, depthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depthTex, 0);
    glGenRenderbuffers(1, &depthRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRBO);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    thickFBO = makeR32FFBO(w, h, thickTex);
    packFBO = makeRGBA32FFBO(w, h, packTex);
    blurFBO_A = makeRGBA32FFBO(w, h, blurTexA);
    blurFBO_B = makeRGBA32FFBO(w, h, blurTexB);

    glGenTextures(1, &sceneTex);
    glBindTexture(GL_TEXTURE_2D, sceneTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

// ─────────────────────────────────────────────────────────────────────────────
inline void FluidRenderer::passDepth(GLuint vao, int n,
    const glm::mat4& proj, const glm::mat4& view)
{
    glBindFramebuffer(GL_FRAMEBUFFER, depthFBO);
    glViewport(0, 0, width, height);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDisable(GL_BLEND);
    glUseProgram(depthProg);
    glUniformMatrix4fv(u.depth_uProj, 1, GL_FALSE, glm::value_ptr(proj));
    glUniformMatrix4fv(u.depth_uView, 1, GL_FALSE, glm::value_ptr(view));
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, n * 3);
    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

inline void FluidRenderer::passThickness(GLuint vao, int n,
    const glm::mat4& proj, const glm::mat4& view)
{
    glBindFramebuffer(GL_FRAMEBUFFER, thickFBO);
    glViewport(0, 0, width, height);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glUseProgram(thickProg);
    glUniformMatrix4fv(u.thick_uProj, 1, GL_FALSE, glm::value_ptr(proj));
    glUniformMatrix4fv(u.thick_uView, 1, GL_FALSE, glm::value_ptr(view));
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, n * 3);
    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

inline void FluidRenderer::passPack()
{
    glBindFramebuffer(GL_FRAMEBUFFER, packFBO);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glUseProgram(packProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, depthTex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, thickTex);
    glUniform1i(u.pack_uDepthTex, 0);
    glUniform1i(u.pack_uThickTex, 1);
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

inline void FluidRenderer::captureSceneColor()
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, sceneTex);
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, width, height);
}

// Single separable blur pass — src (RGBA32F) → dst (RGBA32F)
inline void FluidRenderer::runBlurPass(GLuint srcTex, GLuint dstFBO,
    float dx, float dy, float projScale)
{
    glBindFramebuffer(GL_FRAMEBUFFER, dstFBO);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glUseProgram(blurProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, srcTex);
    glUniform1i(u.blur_uPackTex, 0);
    glUniform2f(u.blur_uTexelSize, 1.0f / width, 1.0f / height);
    glUniform2f(u.blur_uBlurDir, dx, dy);
    glUniform1f(u.blur_uBlurWorldRadius, settings.blurWorldRadius);
    glUniform1f(u.blur_uBlurStrength, settings.blurStrength);
    glUniform1f(u.blur_uBlurDiffStrength, settings.blurDiffStrength);
    glUniform1f(u.blur_uBlurParticleRadius, settings.size);
    glUniform1i(u.blur_uBlurMaxRadius, settings.blurMaxRadius);
    glUniform1f(u.blur_uProjScale, projScale);
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// Pack then 4 blur passes (2 H+V iterations).  Final result in blurTexB.
//   packTex →H→ blurTexA →V→ blurTexB →H→ blurTexA →V→ blurTexB
inline void FluidRenderer::passBlur(const glm::mat4& proj)
{
    // projScale = (screenWidth * P[0][0]) / 2
    // glm column-major: proj[col][row], so proj[0][0] = element (0,0) of perspective matrix
    float projM00 = proj[0][0];
    float projScale = (float)width * projM00 * 0.5f;

    passPack();
    runBlurPass(packTex, blurFBO_A, 1, 0, projScale);  // it1 H
    runBlurPass(blurTexA, blurFBO_B, 0, 1, projScale); // it1 V
    runBlurPass(blurTexB, blurFBO_A, 1, 0, projScale); // it2 H
    runBlurPass(blurTexA, blurFBO_B, 0, 1, projScale); // it2 V  ← final in blurTexB
}

inline void FluidRenderer::passComposite(const glm::vec3& lightDirView,
    float fovDeg, float aspect,
    const glm::mat4& proj)
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, width, height);
    captureSceneColor();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    float tanHalf = tanf(glm::radians(fovDeg * 0.5f));

    glUseProgram(compositeProg);

    // blurTexB = RGBA32F packed (smoothDepth, smoothThick, 0, hardDepth)
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, blurTexB);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, sceneTex);

    glUniform1i(u.comp_uPackTex, 0);
    glUniform1i(u.comp_uSceneTex, 2);
    glUniform2f(u.comp_uTexelSize, 1.0f / width, 1.0f / height);
    glUniform3fv(u.comp_uLightDirView, 1, glm::value_ptr(lightDirView));
    glUniform1f(u.comp_uAbsorption, settings.absorption);
    glUniform3f(u.comp_uExtinction,
        settings.extinctionR, settings.extinctionG, settings.extinctionB);
    glUniform1f(u.comp_uTanHalfFov, tanHalf);
    glUniform1f(u.comp_uAspect, aspect);
    glUniformMatrix4fv(u.comp_uProj, 1, GL_FALSE, glm::value_ptr(proj));
    glUniform1f(u.comp_uRefrMult, settings.refrMult);
    glUniform3f(u.comp_uWaterShallow,
        settings.shallowColorR / 255.f, settings.shallowColorG / 255.f, settings.shallowColorB / 255.f);
    glUniform3f(u.comp_uWaterDeep,
        settings.deepColorR / 255.f, settings.deepColorG / 255.f, settings.deepColorB / 255.f);


    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glUseProgram(0);
}

inline bool FluidRenderer::render(GLuint particleVAO, int particleCount,
    const glm::mat4& proj, const glm::mat4& view,
    int shaderType,
    const glm::vec3& lightDirWorld,
    float fovDegrees, float aspect)
{
    // Legacy particles: return false so drawAll() uses the original program
    if (shaderType == 1)
    {
        settings.heateffect = true;
        return false;
    }
    if (!ready || particleCount <= 0)
        return false;

    // Heat colouring is invisible under the continuous SSF surface
    settings.heateffect = false;

    // Light direction in view space
    glm::vec3 lightDirView = glm::normalize(
        glm::vec3(view * glm::vec4(lightDirWorld, 0.0f)));

    passDepth(particleVAO, particleCount, proj, view);
    passThickness(particleVAO, particleCount, proj, view);
    passBlur(proj);
    passComposite(lightDirView, fovDegrees, aspect, proj);
    return true;
}

inline void FluidRenderer::cleanup()
{
    destroyBuffers();
    if (quadVAO)
    {
        glDeleteVertexArrays(1, &quadVAO);
        quadVAO = 0;
    }
    if (quadVBO)
    {
        glDeleteBuffers(1, &quadVBO);
        quadVBO = 0;
    }
    GLuint progs[] = { depthProg, thickProg, packProg, blurProg, compositeProg };
    for (GLuint p : progs)
        if (p)
            glDeleteProgram(p);
    depthProg = thickProg = packProg = blurProg = compositeProg = 0;
    ready = false;
}