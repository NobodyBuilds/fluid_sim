#pragma once
// ═══════════════════════════════════════════════════════════════════════════════
//  fluid_renderer.h  v4
//
//  shaderType 0 → Screen-space water    (Mode 0 — rewritten v4, Lague reference)
//  shaderType 1 → Legacy particles      (Mode 1 — UNTOUCHED, not one line changed)
//
//  Mode 0 pipeline (6 passes, Lague-faithful, rasterization-only):
//    Pass 1  Depth        sphere billboard → min-depth R32F + gl_FragDepth
//    Pass 2  Thickness    additive constant contribution (Lague: flat 0.1/px)
//    Pass 3  Pack         merge depth+thick → RGBA32F (d, t, t, d)
//    Pass 4  Blur ×N      separable Gaussian on RGBA32F, smoothMask=(1,1,0,0)
//    Pass 5  Normals      min-delta finite diff → world-space RGBA32F
//    Pass 6  Composite    Lague shading model (see below)
//
//  Shading model (matched to Lague FluidRender.shader + NormalsFromDepth.shader):
//    • Full Fresnel equations — IOR 1.0 / 1.33 (Lague's CalculateReflectance)
//      NOT Schlick — full perpendicular + parallel polarisation average
//    • Half-lambert diffuse shading: dot(worldN, dirToSun) * 0.5 + 0.5
//    • Reflection: world-space reflect dir → procedural sky (Lague's SampleSky)
//      Sky: ground / horizon / zenith gradient + tight sun highlight
//      Sun highlight embedded in SampleSky — no separate specular term needed
//    • Beer-Lambert per-channel: transmission = exp(-thickness * extinction_rgb)
//    • Refracted colour: body colour (shallow/deep mix) × transmission × shading
//      NOTE: Lague traces an exact refract ray to an exit point then calls
//      SampleEnvironmentAA — that requires ray-box intersection against scene
//      geometry, which is NOT portable to rasterization. A body-colour
//      approximation is used instead. All other shading terms are identical.
//    • SmoothEdgeNormals: normals at bounding-box edges blend toward face normal
//      (direct port of Lague's SmoothEdgeNormals / CalculateClosestFaceNormal)
//    • Normal reconstruction: per-axis min-delta (forward vs backward finite
//      diff, pick whichever has smaller |Δz|), then cross(ddx,ddy) in OpenGL
//      right-handed view space (Lague uses cross(ddy,ddx) in Unity left-handed)
//
//  Deviations from Lague (all flagged with // NOTE: in shader source):
//    1. Refraction is a body-colour approximation, not environment ray-cast
//    2. cross(ddx,ddy) in normal pass (OpenGL RH) vs cross(ddy,ddx) (Unity LH)
//    3. Sky uses Y-up world convention (match Lague) — if your world is Z-up,
//       swap dir.y ↔ dir.z in SSF_COMPOSITE_FRAG::SampleSky
//    4. No foam integration (no foam pipeline in this renderer)
//    5. No shadow map (requires separate shadow camera — add as TODO if needed)
//    6. Thickness pass keeps no depth test (Lague ZTest LEqual) — minimal visual
//       difference for single-fluid sim with no other opaque geometry
//
//  settings.h additions required:
//    // Gaussian blur (replaces bilateral blurSigma / blurDepthFall)
//    float gaussSigma      = 4.0f;   // Gaussian σ — higher = smoother but slower
//    int   gaussRadius     = 8;      // tap radius; 17 taps per axis per pass
//    int   gaussIterations = 2;      // H+V pairs; 2 = 4 passes total
//    // Sky (Y-up world, match Lague defaults)
//    float skyZenithR=0.08f,  skyZenithG=0.37f,  skyZenithB=0.73f;
//    float skyHorizonR=1.0f,  skyHorizonG=1.0f,  skyHorizonB=1.0f;
//    float skyGroundR=0.186f, skyGroundG=0.159f, skyGroundB=0.186f;
//    float sunIntensity = 8.0f;      // sun highlight brightness multiplier
//    float sunInvSize   = 200.0f;    // sun angular sharpness (pow exponent)
//    // Beer-Lambert extinction per channel (replaces scalar absorption)
//    float extinctionR = 0.45f, extinctionG = 0.18f, extinctionB = 0.08f;
//    // Water body colours
//    float shallowColorR=0.0f,  shallowColorG=0.5f,  shallowColorB=0.8f;
//    float deepColorR=0.0f,     deepColorG=0.1f,     deepColorB=0.3f;
//    // Fluid simulation bounding box (set each frame from sim scale)
//    float boundsSizeX=5.0f, boundsSizeY=5.0f, boundsSizeZ=5.0f;
//    // reflStrength — already exists, repurposed as sky reflectance multiplier
//
//  Performance notes (Mode 0):
//    • All uniform locations cached once at init() — zero glGetUniformLocation/frame
//    • 4 FBOs: depthFBO (R32F+depth), thickFBO (R32F), compFBO (RGBA32F),
//              normalFBO (RGBA32F) — plus 2 ping-pong RGBA32F blur FBOs
//    • Single blur program shared across all blur passes
//    • packProg / normalProg add 2 extra full-screen passes but are trivial cost
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
//  PASS 2 — THICKNESS  (Lague reference: ParticleThickness.shader)
//
//  Flat 0.1 contribution per pixel within the billboard circle.
//  Rendered additively; no depth test (see deviation note 6 in header).
//
//  NOTE: The previous v3 used sqrt(1-r2)*radius*0.030 (chord-based, thicker at
//  centre). Lague uses a constant 0.1 regardless of position within the disc.
//  Both accumulate additively. The flat approach is less physically motivated
//  but matches Lague's ParticleThickness.shader exactly. Tune thicknessParticleScale
//  (billboard radius) and extinctionR/G/B to compensate for the different range.
// ─────────────────────────────────────────────────────────────────────────────
static const char* SSF_THICK_FRAG = R"glsl(
#version 330 core
in vec2  vOffset;
in float vRadius;
out float outThickness;
void main(){
    float r2 = dot(vOffset,vOffset);
    if(r2 >= 1.0) discard;
    // NOTE: Lague's ParticleThickness.shader outputs a constant 0.1 contribution
    // for every pixel within the billboard circle, independent of r2 or vRadius.
    // Reference: "const float contribution = 0.1; return contribution;"
    outThickness = 0.1;
}
)glsl";

// ─────────────────────────────────────────────────────────────────────────────
//  FULLSCREEN QUAD VERTEX  (pack + blur + normal + composite)
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
//  PASS 3 — PACK  (Lague reference: SmoothThickPrepare.shader)
//
//  Merges the R32F depth texture and R32F thickness texture into a single
//  RGBA32F texture. Channel layout:
//    R = depth   (raw, will be blurred → becomes depthSmooth)
//    G = thick   (raw, will be blurred → becomes thickSmooth)
//    B = thick   (preserved through blur via smoothMask.z=0 → thick_hard)
//    A = depth   (preserved through blur via smoothMask.w=0 → depth_hard)
//
//  This layout is byte-for-byte identical to Lague's SmoothThickPrepare.shader:
//    "return float4(depth, thick, thick, depth);"
// ─────────────────────────────────────────────────────────────────────────────
static const char* SSF_PACK_FRAG = R"glsl(
#version 330 core
uniform sampler2D uDepthTex;   // R32F — raw linear sphere depth (0 = background)
uniform sampler2D uThickTex;   // R32F — additive thickness accumulation
in  vec2 vUV;
out vec4 outPacked;
void main(){
    float depth = texture(uDepthTex, vUV).r;
    float thick = texture(uThickTex, vUV).r;
    // Pack: (depth, thick, thick, depth)
    // R and G will be Gaussian-blurred; B and A preserved (smoothMask=(1,1,0,0)).
    outPacked = vec4(depth, thick, thick, depth);
}
)glsl";

// ─────────────────────────────────────────────────────────────────────────────
//  PASS 4 — SEPARABLE GAUSSIAN BLUR  (Lague reference: GaussPass.hlsl)
//
//  Input / output: RGBA32F packed texture (d, t, t, d layout from pass 3).
//  Blurs R and G channels (smoothMask.x=1, smoothMask.y=1).
//  B and A are passed through unchanged — they hold the unsmoothed originals.
//
//  Output channel layout after N H+V iteration pairs:
//    R = depthSmooth     (blurred depth  — used for normal reconstruction)
//    G = thickSmooth     (blurred thick  — used for Beer-Lambert & body colour)
//    B = thick_hard      (raw thick      — available for debug display)
//    A = depth_hard      (raw depth      — background guard + debug display)
//
//  Background pixels (A < 0.001) are skipped as sample contributors so blur
//  does not bleed the fluid silhouette into the surrounding background.
//  The output A is always the centre pixel's original hard depth (not blurred).
//
//  Gaussian kernel: w(i) = exp(-i² / (2σ²))  — matches GaussPass.hlsl
//    "float Calculate1DGaussianKernel(int x, float sigma) {
//       float c = 2 * sigma * sigma; return exp(-x * x / c); }"
// ─────────────────────────────────────────────────────────────────────────────
static const char* SSF_BLUR_FRAG = R"glsl(
#version 330 core
uniform sampler2D uTex;         // RGBA32F packed (R=depth, G=thick, B=thick, A=depth)
uniform vec2      uTexelSize;
uniform vec2      uBlurDir;     // (1,0) horizontal pass | (0,1) vertical pass
uniform float     uSigma;       // Gaussian sigma (world-space or screen-space)
uniform int       uRadius;      // tap radius; total taps = 2*radius+1
uniform vec3      uSmoothMask;  // per-channel: 1=blur, 0=keep original (use (1,1,0))
in  vec2 vUV;
out vec4 outBlurred;

float GaussKernel(int x, float sigma){
    float c = 2.0 * sigma * sigma;
    return exp(-float(x * x) / c);
}

void main(){
    vec4  original  = texture(uTex, vUV);
    float hardDepth = original.w;   // A = unblurred depth; used as background guard

    // Background pixel — no fluid here; pass through unchanged
    if(hardDepth < 0.001){ outBlurred = original; return; }

    vec4  sum  = vec4(0.0);
    float wSum = 0.0;

    for(int i = -uRadius; i <= uRadius; ++i){
        vec2 sampleUV = vUV + uBlurDir * float(i) * uTexelSize;
        vec4 s = texture(uTex, sampleUV);
        // Skip background samples — prevents silhouette halo bleeding.
        // GaussPass.hlsl: "if (depth < 10000)" — our background depth is 0 so
        // check A > 0.001 instead (equivalent guard, different sentinel convention).
        if(s.a > 0.001){
            float w = GaussKernel(i, uSigma);
            sum  += s * w;
            wSum += w;
        }
    }

    vec3 blurredRGB = (wSum > 1e-6) ? (sum / wSum).rgb : original.rgb;

    // smoothMask lerp: channels with mask=1 get blurred result, mask=0 keep original.
    // GaussPass.hlsl: "return float4(lerp(original.rgb, blurResult, smoothMask), depth);"
    vec3 outRGB = mix(original.rgb, blurredRGB, uSmoothMask);

    // A is always the centre pixel's original hard depth — never blurred.
    outBlurred = vec4(outRGB, hardDepth);
}
)glsl";

// ─────────────────────────────────────────────────────────────────────────────
//  PASS 5 — NORMAL RECONSTRUCTION  (Lague reference: NormalsFromDepth.shader)
//
//  Input:  RGBA32F packed texture after blur (R=depthSmooth, G=thickSmooth,
//          B=thick_hard, A=depth_hard)
//  Output: RGBA32F world-space normal (XYZ=normal, W=1; W=0 for background)
//
//  Algorithm (direct port of NormalsFromDepth.shader):
//    Per axis, compute both forward and backward finite differences.
//    Choose whichever has the smaller |Δz| — this avoids normal smearing at
//    depth discontinuities (surface edges against background or other objects).
//    "float3 ddx  = ViewPos(uv + float2(o.x, 0)) - posCentre;
//     float3 ddx2 = posCentre - ViewPos(uv - float2(o.x, 0));
//     if (abs(ddx2.z) < abs(ddx.z)) ddx = ddx2;"
//
//  NOTE: Lague uses cross(ddy, ddx) in Unity left-handed view space (+Z toward
//  camera). In OpenGL right-handed view space (+Z toward camera, -Z into scene),
//  the same winding gives N.z < 0 (away from camera) for a flat surface.
//  We use cross(ddx, ddy) instead and add a N.z < 0 flip guard to guarantee
//  the outward-facing normal in OpenGL view space.
//
//  NOTE: Lague's ViewPos uses unity_CameraInvProjection. We reconstruct from
//  uTanHalfFov + uAspect — equivalent for a standard pinhole perspective camera.
//  ViewPos: pos = vec3(ndc * vec2(aspect,1) * tanHalfFov * d, -d)  (OpenGL RH)
// ─────────────────────────────────────────────────────────────────────────────
static const char* SSF_NORMAL_FRAG = R"glsl(
#version 330 core
uniform sampler2D uCompTex;    // RGBA32F after blur (R=depthSmooth, A=depth_hard)
uniform vec2      uTexelSize;
uniform float     uTanHalfFov;
uniform float     uAspect;
uniform mat3      uViewRotInv; // transpose(mat3(view)) — transforms view→world
uniform int       uUseSmoothed;// 1 = read R (smoothed depth); 0 = read A (hard depth)
in  vec2 vUV;
out vec4 outNormal;            // XYZ = world-space normal, W = 1 (0 = background)

// Reconstruct view-space position from screen UV and positive linear depth.
// OpenGL right-handed: camera at origin, scene along -Z, so pos.z = -d.
vec3 ViewPos(vec2 uv, float d){
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(ndc * vec2(uAspect, 1.0) * uTanHalfFov * d, -d);
}

float SampleDepth(vec2 uv){
    vec4 p = texture(uCompTex, uv);
    return (uUseSmoothed == 1) ? p.r : p.a;
}

void main(){
    float depthC = SampleDepth(vUV);

    // Background: no fluid at this pixel
    if(depthC < 0.001){ outNormal = vec4(0.0); return; }

    vec3 posC = ViewPos(vUV, depthC);

    // ── Min-delta finite differences (Lague's NormalsFromDepth.shader) ─────────
    float dR = SampleDepth(vUV + vec2(uTexelSize.x, 0.0));
    float dL = SampleDepth(vUV - vec2(uTexelSize.x, 0.0));
    vec3 ddx  = ViewPos(vUV + vec2(uTexelSize.x,0.0), dR) - posC;
    vec3 ddx2 = posC - ViewPos(vUV - vec2(uTexelSize.x,0.0), dL);
    if(abs(ddx2.z) < abs(ddx.z)) ddx = ddx2;

    float dU = SampleDepth(vUV + vec2(0.0, uTexelSize.y));
    float dD = SampleDepth(vUV - vec2(0.0, uTexelSize.y));
    vec3 ddy  = ViewPos(vUV + vec2(0.0, uTexelSize.y), dU) - posC;
    vec3 ddy2 = posC - ViewPos(vUV - vec2(0.0, uTexelSize.y), dD);
    if(abs(ddy2.z) < abs(ddy.z)) ddy = ddy2;

    // NOTE: Lague: cross(ddy, ddx) — correct for Unity LH view space (+Z toward cam).
    // OpenGL RH view space: cross(ddx, ddy) gives the outward-facing normal (N.z > 0).
    // Flip guard added for robustness at silhouette edges.
    vec3 viewNormal = cross(ddx, ddy);
    float lenN = length(viewNormal);
    if(lenN < 1e-8){ outNormal = vec4(0.0); return; } // degenerate — both neighbours background
    viewNormal = viewNormal / lenN;
    if(viewNormal.z < 0.0) viewNormal = -viewNormal;   // ensure outward in OpenGL RH

    // Transform view-space normal to world space via view rotation inverse.
    // viewRotInv = transpose(mat3(view)); valid because view rotation is orthonormal.
    // Lague: "float3 worldNormal = mul(unity_CameraToWorld, float4(viewNormal, 0));"
    vec3 worldNormal = normalize(uViewRotInv * viewNormal);

    outNormal = vec4(worldNormal, 1.0);
}
)glsl";

// ─────────────────────────────────────────────────────────────────────────────
//  PASS 6 — COMPOSITE  (Lague reference: FluidRender.shader frag + all helpers)
//
//  Inputs:
//    uCompTex   — RGBA32F (R=depthSmooth, G=thickSmooth, B=thick_hard, A=depth_hard)
//    uNormalTex — RGBA32F world-space normals (XYZ, W=0 for background)
//
//  Implements the full Lague shading pipeline:
//    CalculateReflectance  — full Fresnel, not Schlick
//    SampleSky             — ground / horizon / zenith + sun highlight
//    SmoothEdgeNormals     — normal blending at bounding box edges
//    Half-lambert shading  — dot(N, dirToSun) * 0.5 + 0.5
//    Beer-Lambert          — transmission = exp(-thickness * extinction_rgb)
//
//  Deviations from Lague (rasterization-only constraints):
//    REFRACTION: Lague traces refractDir to an exit point then calls
//    SampleEnvironmentAA. That requires ray-box intersection — not possible
//    in a rasterization pass without a background depth+colour buffer.
//    Replaced with: body colour (shallow/deep mix) × transmission × shading.
//    All other shading terms are identical to Lague.
//
//    SKY BACKGROUND: Lague returns SampleEnvironmentAA for non-fluid pixels.
//    We return SampleSky(viewDirWorld) instead (no floor geometry available).
//
//    FOAM: Lague integrates a foam layer. Not present here.
//    SHADOW MAP: Lague samples a shadow RT for floor lit colour. Not present here.
// ─────────────────────────────────────────────────────────────────────────────
static const char* SSF_COMPOSITE_FRAG = R"glsl(
#version 330 core
uniform sampler2D uCompTex;      // RGBA32F: R=depthSmooth G=thickSmooth B=thick_hard A=depth_hard
uniform sampler2D uNormalTex;    // RGBA32F: XYZ=worldNormal W=1(fluid)/0(bg)

uniform vec3  uDirToSun;         // world-space, normalised, pointing TOWARD sun
uniform mat3  uViewRotInv;       // transpose(mat3(view)) — view-space → world-space
uniform vec3  uCamPosWorld;      // camera world position (for surface hit reconstruction)

// Beer-Lambert extinction (per channel, replaces scalar uAbsorption from v3)
uniform vec3  uExtinction;       // (k_r, k_g, k_b) — higher = more absorbed

// Water body colours (thin surface vs deep interior)
uniform vec3  uShallowColor;
uniform vec3  uDeepColor;

// Sky gradient (Y-up world — matches Lague's convention)
uniform vec3  uSkyZenith;        // colour directly overhead
uniform vec3  uSkyHorizon;       // colour at horizon
uniform vec3  uSkyGround;        // colour below horizon
uniform float uSunIntensity;     // sun highlight brightness multiplier
uniform float uSunInvSize;       // sun sharpness (pow exponent — higher = tighter disc)
uniform float uReflStrength;     // overall reflection colour multiplier

// Fluid simulation bounding box (world space, centred at origin)
uniform vec3  uBoundsSize;       // used by SmoothEdgeNormals

// Projection parameters (for view-dir reconstruction)
uniform float uTanHalfFov;
uniform float uAspect;

in  vec2 vUV;
out vec4 outColor;

// ── Sky (Lague's SampleSky, Y-up world) ──────────────────────────────────────
// NOTE: Lague's world is Y-up. This function uses dir.y for the vertical axis.
// If your app is Z-up, swap dir.y for dir.z in skyGradientT, groundToSkyT, and
// the step() sun guard below.
vec3 SampleSky(vec3 dir){
    // "float sun = pow(max(0, dot(dir, dirToSun)), sunInvSize) * sunIntensity;"
    float sun          = pow(max(0.0, dot(dir, uDirToSun)), uSunInvSize) * uSunIntensity;
    float skyGradientT = pow(smoothstep(0.0, 0.4, dir.y), 0.35);
    float groundToSkyT = smoothstep(-0.01, 0.0, dir.y);
    vec3  skyGradient  = mix(uSkyHorizon, uSkyZenith, skyGradientT);
    // Sun only added above horizon: Lague "sun * (groundToSkyT >= 1)"
    // In GLSL, bool → float comparison is unavailable; use step(0.9999, x) instead.
    float sunMask = (groundToSkyT > 0.999) ? 1.0 : 0.0;
return mix(uSkyGround, skyGradient, groundToSkyT) + (sun * sunMask);
}

// ── Full Fresnel (Lague's CalculateReflectance) ───────────────────────────────
// Accounts for both perpendicular and parallel polarisation.
// Lague: "return (rPerpendicular + rParallel) / 2"
// This is more accurate than Schlick and matches Lague's exact implementation.
float CalculateReflectance(vec3 inDir, vec3 normal, float iorA, float iorB){
    float refractRatio        = iorA / iorB;
    float cosAngleIn          = -dot(inDir, normal);
    float sinSqrAngleRefract  = refractRatio * refractRatio * (1.0 - cosAngleIn * cosAngleIn);
    if(sinSqrAngleRefract >= 1.0) return 1.0;   // total internal reflection

    float cosAngleRefract = sqrt(max(0.0, 1.0 - sinSqrAngleRefract));
    float rPerp = (iorA * cosAngleIn - iorB * cosAngleRefract) /
                  (iorA * cosAngleIn + iorB * cosAngleRefract);
    rPerp *= rPerp;
    float rPara = (iorB * cosAngleIn - iorA * cosAngleRefract) /
                  (iorB * cosAngleIn + iorA * cosAngleRefract);
    rPara *= rPara;
    return (rPerp + rPara) * 0.5;
}

// ── Closest face normal on a box (Lague's CalculateClosestFaceNormal) ─────────
// Assumes box centred at world origin. boundsSize = full extents (not half).
// NOTE: Assumes fluid sim is centred at world origin. If sim.transform.position != 0,
// subtract that offset from pos before calling this function.
vec3 ClosestFaceNormal(vec3 pos){
    vec3 halfSize = uBoundsSize * 0.5;
    vec3 o = halfSize - abs(pos);
    if(o.x < o.y && o.x < o.z) return vec3(sign(pos.x), 0.0, 0.0);
    if(o.y < o.z)               return vec3(0.0, sign(pos.y), 0.0);
    return                             vec3(0.0, 0.0, sign(pos.z));
}

// ── Smooth edge normals (Lague's SmoothEdgeNormals, direct port) ──────────────
// Blends the fluid surface normal toward the nearest box face normal at edges.
// Prevents the seam artifact where fluid normals point "out through the wall."
// Lague: "faceWeight = 1 - smoothstep(0, smoothDst, faceWeight);"
//        "normal = normalize(normal + smoothEdgeNormal * 6 * max(0, dot(...)));"
vec3 SmoothEdgeNormals(vec3 N, vec3 pos){
    const float smoothDst = 0.01;
    vec3 o = uBoundsSize * 0.5 - abs(pos);
    // faceWeight: distance to closest XZ edge (Lague uses min(o.x, o.z), not o.y)
    float faceWeight  = max(0.0, min(o.x, o.z));
    vec3  faceNormal  = ClosestFaceNormal(pos);
    float cornerWeight = 1.0 - clamp(abs(o.x - o.z) * 6.0, 0.0, 1.0);
    faceWeight = 1.0 - smoothstep(0.0, smoothDst, faceWeight);
    faceWeight *= (1.0 - cornerWeight);
    return normalize(N * (1.0 - faceWeight) + faceNormal * faceWeight);
}

void main(){
    vec4  packed     = texture(uCompTex, vUV);
    float depthSmooth  = packed.r;
    float thickness    = packed.g;   // smoothed — Beer-Lambert, body colour mix
    float thickness_hard = packed.b; // raw — debug only
    float depth_hard   = packed.a;

if(depthSmooth < 0.001){
        vec2 ndc = vUV * 2.0 - 1.0;
        vec3 vdView = normalize(vec3(ndc * vec2(uAspect, 1.0) * uTanHalfFov, -1.0));
        vec3 vdWorld = normalize(uViewRotInv * vdView);
        outColor = vec4(SampleSky(vdWorld), 1.0);
        return;
    }

    // ── Reconstruct world-space view direction ────────────────────────────────
    // Lague: "float3 viewDirWorld = WorldViewDir(uv)" — Unity CameraInvProjection.
    // NOTE: We reconstruct from uTanHalfFov + uAspect; equivalent for pinhole camera.
    vec2 ndc           = vUV * 2.0 - 1.0;
    vec3 viewDirView   = normalize(vec3(ndc * vec2(uAspect, 1.0) * uTanHalfFov, -1.0));
    vec3 viewDirWorld  = normalize(uViewRotInv * viewDirView);

    // ── Background: no fluid surface at this pixel ────────────────────────────
    // Lague: "if (depthSmooth > 1000) return float4(world, 1) * (1-foam) + foam;"
    // NOTE: Our background depth is 0 (cleared to black), not 10^7.
    // We return the sky colour for non-fluid pixels instead of a full environment.
    // This replaces Lague's SampleEnvironmentAA (floor ray-cast not available).
    if(depthSmooth < 0.001){
        outColor = vec4(SampleSky(viewDirWorld), 1.0);
        return;
    }

    // ── World-space normal (from normal pass) ─────────────────────────────────
    vec4  normalSample = texture(uNormalTex, vUV);
    // W = 0 means normal reconstruction failed (both neighbours background)
    if(normalSample.w < 0.5){ outColor = vec4(SampleSky(viewDirWorld), 1.0); return; }
    vec3 normal = normalize(normalSample.xyz);

    // ── Surface hit point in world space ─────────────────────────────────────
    // Lague: "float3 hitPos = _WorldSpaceCameraPos + viewDirWorld * depthSmooth;"
    // depthSmooth is positive linear depth from camera; viewDirWorld is normalised.
    vec3 hitPos = uCamPosWorld + viewDirWorld * depthSmooth;

    // ── Smooth edge normals (Lague, direct port) ──────────────────────────────
    // Lague: "float3 smoothEdgeNormal = SmoothEdgeNormals(normal, hitPos, boundsSize);"
    //        "normal = normalize(normal + smoothEdgeNormal * 6 * max(0, dot(normal, smoothEdgeNormal)));"
    vec3 smoothEdge = SmoothEdgeNormals(normal, hitPos);
    normal = normalize(normal + smoothEdge * 6.0 * max(0.0, dot(normal, smoothEdge)));

    // ── Half-lambert shading (Lague) ──────────────────────────────────────────
    // "float shading = dot(normal, dirToSun) * 0.5 + 0.5;"
    // "shading = shading * (1 - ambientLight) + ambientLight;"
    const float ambientLight = 0.3;
    float shading = dot(normal, uDirToSun) * 0.5 + 0.5;
    shading = shading * (1.0 - ambientLight) + ambientLight;

    // ── Full Fresnel (Lague's CalculateReflectionAndRefraction) ──────────────
    // "LightResponse lightResponse = CalculateReflectionAndRefraction(viewDirWorld, normal, 1, 1.33);"
    // viewDirWorld points FROM camera TOWARD surface; normal points AWAY from surface.
    float reflectWeight = CalculateReflectance(viewDirWorld, normal, 1.0, 1.33);
    float refractWeight = 1.0 - reflectWeight;

    // ── Reflection colour: sky sampled at reflected world-space direction ──────
    // Lague: "float3 reflectDir = reflect(viewDirWorld, normal);"
    //        "float3 reflectCol = SampleEnvironmentAA(hitPos, lightResponse.reflectDir);"
    // SampleEnvironmentAA includes sky + floor. We only have sky (rasterization-only).
    // The sun highlight is embedded in SampleSky, so it is included here automatically.
    vec3 reflectDir = reflect(viewDirWorld, normal);
    vec3 reflectCol = SampleSky(reflectDir) * uReflStrength;

    // ── Beer-Lambert transmission per channel (Lague) ─────────────────────────
    // Lague: "float3 transmission = exp(-thickness * extinctionCoefficients);"
    // extinctionCoefficients is set per-channel via FluidRenderTest.cs as a vec3.
    vec3 transmission = exp(-thickness * uExtinction);

    // ── Refraction colour (approximation — see header NOTE) ───────────────────
    // Lague: "float3 exitPos = hitPos + lightResponse.refractDir * thickness * refractionMultiplier;"
    //        "float3 refractCol = SampleEnvironmentAA(exitPos, viewDirWorld);"
    //        "refractCol *= transmission;"
    // NOTE: The exact exit-point calculation requires a refraction ray direction
    // (computable) and then a ray-box intersection to find what the ray hits —
    // which is the floor/environment. This is not available in rasterization-only.
    // We substitute: mix of deep/shallow body colour, attenuated by transmission
    // and modulated by half-lambert shading (equivalent ambient+diffuse contribution).
    float thinness = clamp(1.0 - thickness * 0.4, 0.0, 1.0);
    vec3  bodyColor = mix(uDeepColor, uShallowColor, thinness);
    vec3  refractCol = bodyColor * transmission * shading;

    // ── Blend reflected and refracted colour (Lague) ─────────────────────────
    // Lague: "float3 col = lerp(reflectCol, refractCol, lightResponse.refractWeight);"
    // reflectWeight drives toward mirror reflection; refractWeight toward fluid body.
    vec3 col = mix(refractCol, reflectCol, reflectWeight);

    outColor = vec4(col, 1.0);
}
)glsl";

// ═══════════════════════════════════════════════════════════════════════════════
//  Cached uniform locations — populated once in cacheUniforms(), used every frame.
//  Zero glGetUniformLocation calls in the render hot path.
// ═══════════════════════════════════════════════════════════════════════════════
struct SSFUniforms {
    // ── Pass 1: depth ─────────────────────────────────────────────────────────
    GLint depth_uProj = -1, depth_uView = -1;
    // ── Pass 2: thickness ─────────────────────────────────────────────────────
    GLint thick_uProj = -1, thick_uView = -1;
    // ── Pass 3: pack ──────────────────────────────────────────────────────────
    GLint pack_uDepthTex = -1, pack_uThickTex = -1;
    // ── Pass 4: Gaussian blur (all 2*N passes share one program) ──────────────
    GLint blur_uTex = -1, blur_uTexelSize = -1, blur_uBlurDir = -1;
    GLint blur_uSigma = -1, blur_uRadius = -1, blur_uSmoothMask = -1;
    // ── Pass 5: normal reconstruction ─────────────────────────────────────────
    GLint norm_uCompTex = -1, norm_uTexelSize = -1;
    GLint norm_uTanHalfFov = -1, norm_uAspect = -1;
    GLint norm_uViewRotInv = -1, norm_uUseSmoothed = -1;
    // ── Pass 6: composite ─────────────────────────────────────────────────────
    GLint comp_uCompTex = -1, comp_uNormalTex = -1;
    GLint comp_uDirToSun = -1, comp_uViewRotInv = -1, comp_uCamPosWorld = -1;
    GLint comp_uExtinction = -1;
    GLint comp_uShallowColor = -1, comp_uDeepColor = -1;
    GLint comp_uSkyZenith = -1, comp_uSkyHorizon = -1, comp_uSkyGround = -1;
    GLint comp_uSunIntensity = -1, comp_uSunInvSize = -1, comp_uReflStrength = -1;
    GLint comp_uBoundsSize = -1;
    GLint comp_uTanHalfFov = -1, comp_uAspect = -1;
};

// ═══════════════════════════════════════════════════════════════════════════════
//  FluidRenderer
// ═══════════════════════════════════════════════════════════════════════════════
struct FluidRenderer {
    // ── FBOs ──────────────────────────────────────────────────────────────────
    GLuint depthFBO = 0;  // Pass 1: R32F colour + hardware depth RBO
    GLuint thickFBO = 0;  // Pass 2: R32F additive thickness
    GLuint compFBO = 0;  // Pass 3: RGBA32F packed (d,t,t,d) — blur source
    GLuint blurFBO_A = 0;  // Pass 4 ping: RGBA32F
    GLuint blurFBO_B = 0;  // Pass 4 pong: RGBA32F — final blur result
    GLuint normalFBO = 0;  // Pass 5: RGBA32F world-space normals

    // ── Textures ──────────────────────────────────────────────────────────────
    GLuint depthTex = 0;  // R32F sphere depth
    GLuint thickTex = 0;  // R32F additive thickness
    GLuint compTex = 0;  // RGBA32F packed (input to blur)
    GLuint blurTexA = 0;  // RGBA32F blur ping
    GLuint blurTexB = 0;  // RGBA32F blur pong (final blurred result)
    GLuint normalTex = 0;  // RGBA32F world-space normals
    GLuint depthRBO = 0;  // hardware depth renderbuffer (Pass 1 self-occlusion)

    // ── Programs ──────────────────────────────────────────────────────────────
    GLuint depthProg = 0;
    GLuint thickProg = 0;
    GLuint packProg = 0;  // new v4
    GLuint blurProg = 0;  // single program shared by all blur passes
    GLuint normalProg = 0;  // new v4
    GLuint compositeProg = 0;

    GLuint quadVAO = 0, quadVBO = 0;
    SSFUniforms u;

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
    GLuint makeRGBA32FFBO(int w, int h, GLuint& texOut);   // new v4
    GLuint compileShader(GLenum type, const char* src);
    GLuint linkProgram(const char* vs, const char* fs);
    void   initQuad();
    void   cacheUniforms();
    void   destroyBuffers();

    void passDepth(GLuint vao, int n, const glm::mat4& proj, const glm::mat4& view);
    void passThickness(GLuint vao, int n, const glm::mat4& proj, const glm::mat4& view);
    void passPack();                                            // new v4
    void runBlurPass(GLuint srcTex, GLuint dstFBO, float dx, float dy);
    void passBlur();
    void passNormal(const glm::mat3& viewRotInv, float tanHalfFov, float aspect);  // new v4
    void passComposite(const glm::vec3& lightDirWorld,
        const glm::mat3& viewRotInv,
        const glm::vec3& camPosWorld,
        float tanHalfFov, float aspect);
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

// R32F FBO — for depth and thickness passes (single-channel floating-point)
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

// RGBA32F FBO — for packed buffer, blur ping-pong, and world-space normals
inline GLuint FluidRenderer::makeRGBA32FFBO(int w, int h, GLuint& texOut) {
    GLuint fbo;
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &texOut);  glBindTexture(GL_TEXTURE_2D, texOut);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
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

inline void FluidRenderer::cacheUniforms() {
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
    u.blur_uTex = glGetUniformLocation(blurProg, "uTex");
    u.blur_uTexelSize = glGetUniformLocation(blurProg, "uTexelSize");
    u.blur_uBlurDir = glGetUniformLocation(blurProg, "uBlurDir");
    u.blur_uSigma = glGetUniformLocation(blurProg, "uSigma");
    u.blur_uRadius = glGetUniformLocation(blurProg, "uRadius");
    u.blur_uSmoothMask = glGetUniformLocation(blurProg, "uSmoothMask");
    // normal
    u.norm_uCompTex = glGetUniformLocation(normalProg, "uCompTex");
    u.norm_uTexelSize = glGetUniformLocation(normalProg, "uTexelSize");
    u.norm_uTanHalfFov = glGetUniformLocation(normalProg, "uTanHalfFov");
    u.norm_uAspect = glGetUniformLocation(normalProg, "uAspect");
    u.norm_uViewRotInv = glGetUniformLocation(normalProg, "uViewRotInv");
    u.norm_uUseSmoothed = glGetUniformLocation(normalProg, "uUseSmoothed");
    // composite
    u.comp_uCompTex = glGetUniformLocation(compositeProg, "uCompTex");
    u.comp_uNormalTex = glGetUniformLocation(compositeProg, "uNormalTex");
    u.comp_uDirToSun = glGetUniformLocation(compositeProg, "uDirToSun");
    u.comp_uViewRotInv = glGetUniformLocation(compositeProg, "uViewRotInv");
    u.comp_uCamPosWorld = glGetUniformLocation(compositeProg, "uCamPosWorld");
    u.comp_uExtinction = glGetUniformLocation(compositeProg, "uExtinction");
    u.comp_uShallowColor = glGetUniformLocation(compositeProg, "uShallowColor");
    u.comp_uDeepColor = glGetUniformLocation(compositeProg, "uDeepColor");
    u.comp_uSkyZenith = glGetUniformLocation(compositeProg, "uSkyZenith");
    u.comp_uSkyHorizon = glGetUniformLocation(compositeProg, "uSkyHorizon");
    u.comp_uSkyGround = glGetUniformLocation(compositeProg, "uSkyGround");
    u.comp_uSunIntensity = glGetUniformLocation(compositeProg, "uSunIntensity");
    u.comp_uSunInvSize = glGetUniformLocation(compositeProg, "uSunInvSize");
    u.comp_uReflStrength = glGetUniformLocation(compositeProg, "uReflStrength");
    u.comp_uBoundsSize = glGetUniformLocation(compositeProg, "uBoundsSize");
    u.comp_uTanHalfFov = glGetUniformLocation(compositeProg, "uTanHalfFov");
    u.comp_uAspect = glGetUniformLocation(compositeProg, "uAspect");
}

inline void FluidRenderer::destroyBuffers() {
    GLuint fbos[] = { depthFBO, thickFBO, compFBO, blurFBO_A, blurFBO_B, normalFBO };
    GLuint texs[] = { depthTex, thickTex, compTex, blurTexA,  blurTexB,  normalTex };
    glDeleteFramebuffers(6, fbos);
    glDeleteTextures(6, texs);
    glDeleteRenderbuffers(1, &depthRBO);
    depthFBO = thickFBO = compFBO = blurFBO_A = blurFBO_B = normalFBO = 0;
    depthTex = thickTex = compTex = blurTexA = blurTexB = normalTex = 0;
    depthRBO = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
inline void FluidRenderer::init(int w, int h) {
    width = w; height = h;

    depthProg = linkProgram(SSF_DEPTH_VERT, SSF_DEPTH_FRAG);
    thickProg = linkProgram(SSF_DEPTH_VERT, SSF_THICK_FRAG);
    packProg = linkProgram(SSF_QUAD_VERT, SSF_PACK_FRAG);
    blurProg = linkProgram(SSF_QUAD_VERT, SSF_BLUR_FRAG);
    normalProg = linkProgram(SSF_QUAD_VERT, SSF_NORMAL_FRAG);
    compositeProg = linkProgram(SSF_QUAD_VERT, SSF_COMPOSITE_FRAG);
    cacheUniforms();

    // ── Pass 1: Depth — R32F colour + hardware depth renderbuffer ────────────
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

    // ── Pass 2: Thickness ────────────────────────────────────────────────────
    thickFBO = makeR32FFBO(w, h, thickTex);

    // ── Pass 3: Pack ─────────────────────────────────────────────────────────
    compFBO = makeRGBA32FFBO(w, h, compTex);

    // ── Pass 4: Blur ping-pong (RGBA32F) ─────────────────────────────────────
    blurFBO_A = makeRGBA32FFBO(w, h, blurTexA);
    blurFBO_B = makeRGBA32FFBO(w, h, blurTexB);

    // ── Pass 5: Normals ───────────────────────────────────────────────────────
    normalFBO = makeRGBA32FFBO(w, h, normalTex);

    initQuad();
    ready = true;
}

// ─────────────────────────────────────────────────────────────────────────────
inline void FluidRenderer::resize(int w, int h) {
    destroyBuffers();
    width = w; height = h;

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

    thickFBO = makeR32FFBO(w, h, thickTex);
    compFBO = makeRGBA32FFBO(w, h, compTex);
    blurFBO_A = makeRGBA32FFBO(w, h, blurTexA);
    blurFBO_B = makeRGBA32FFBO(w, h, blurTexB);
    normalFBO = makeRGBA32FFBO(w, h, normalTex);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pass 1: Depth — sphere billboard → min-depth R32F
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

// ─────────────────────────────────────────────────────────────────────────────
//  Pass 2: Thickness — additive flat 0.1 contribution per pixel
//
//  NOTE: No depth test (see header deviation note 6).
//  Lague's Unity shader uses ZTest LEqual. In a single-fluid sim with no other
//  opaque geometry this makes no observable difference, so we keep the simpler
//  no-depth-test path. To match Lague exactly, attach the depth RBO to thickFBO
//  and call glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL) here.
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
//  Pass 3: Pack — merge depth + thick into RGBA32F (Lague's SmoothThickPrepare)
// ─────────────────────────────────────────────────────────────────────────────
inline void FluidRenderer::passPack() {
    glBindFramebuffer(GL_FRAMEBUFFER, compFBO);
    glViewport(0, 0, width, height); glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND);
    glUseProgram(packProg);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, depthTex);
    glUniform1i(u.pack_uDepthTex, 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, thickTex);
    glUniform1i(u.pack_uThickTex, 1);
    glBindVertexArray(quadVAO); glDrawArrays(GL_TRIANGLES, 0, 6); glBindVertexArray(0);
    glUseProgram(0); glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pass 4 (single pass): Gaussian blur on RGBA32F — src → dst
//  smoothMask=(1,1,0) is set in passBlur() — R and G blurred, B preserved.
// ─────────────────────────────────────────────────────────────────────────────
inline void FluidRenderer::runBlurPass(GLuint srcTex, GLuint dstFBO, float dx, float dy) {
    glBindFramebuffer(GL_FRAMEBUFFER, dstFBO);
    glViewport(0, 0, width, height); glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND);
    glUseProgram(blurProg);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, srcTex);
    glUniform1i(u.blur_uTex, 0);
    glUniform2f(u.blur_uTexelSize, 1.0f / width, 1.0f / height);
    glUniform2f(u.blur_uBlurDir, dx, dy);
    glUniform1f(u.blur_uSigma, settings.gaussSigma);
    glUniform1i(u.blur_uRadius, settings.gaussRadius);
    // smoothMask=(1,1,0): blur R (depth) and G (thickness), preserve B (thick_hard)
    glUniform3f(u.blur_uSmoothMask, 1.0f, 1.0f, 0.0f);
    glBindVertexArray(quadVAO); glDrawArrays(GL_TRIANGLES, 0, 6); glBindVertexArray(0);
    glUseProgram(0); glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// N H+V Gaussian iteration pairs (settings.gaussIterations, default 2 = 4 passes).
// Final result always lands in blurTexB after an even number of passes.
// Iteration 1: compTex →H→ blurTexA →V→ blurTexB
// Iteration k: blurTexB →H→ blurTexA →V→ blurTexB
inline void FluidRenderer::passBlur() {
    // First iteration reads from the packed compTex
    runBlurPass(compTex, blurFBO_A, 1, 0);
    runBlurPass(blurTexA, blurFBO_B, 0, 1);
    // Additional iterations ping-pong between blurTexB → blurTexA → blurTexB
    for (int i = 1; i < settings.gaussIterations; ++i) {
        runBlurPass(blurTexB, blurFBO_A, 1, 0);
        runBlurPass(blurTexA, blurFBO_B, 0, 1);
    }
    // Final result is always in blurTexB
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pass 5: Normal reconstruction — min-delta finite differences → world-space
// ─────────────────────────────────────────────────────────────────────────────
inline void FluidRenderer::passNormal(const glm::mat3& viewRotInv,
    float tanHalfFov, float aspect) {
    glBindFramebuffer(GL_FRAMEBUFFER, normalFBO);
    glViewport(0, 0, width, height); glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND);
    glUseProgram(normalProg);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, blurTexB);
    glUniform1i(u.norm_uCompTex, 0);
    glUniform2f(u.norm_uTexelSize, 1.0f / width, 1.0f / height);
    glUniform1f(u.norm_uTanHalfFov, tanHalfFov);
    glUniform1f(u.norm_uAspect, aspect);
    glUniformMatrix3fv(u.norm_uViewRotInv, 1, GL_FALSE, glm::value_ptr(viewRotInv));
    glUniform1i(u.norm_uUseSmoothed, 1);  // use blurred depth (R channel) for normals
    glBindVertexArray(quadVAO); glDrawArrays(GL_TRIANGLES, 0, 6); glBindVertexArray(0);
    glUseProgram(0); glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pass 6: Composite — Lague shading model, draw to screen
// ─────────────────────────────────────────────────────────────────────────────
inline void FluidRenderer::passComposite(const glm::vec3& lightDirWorld,
    const glm::mat3& viewRotInv,
    const glm::vec3& camPosWorld,
    float tanHalfFov, float aspect) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, width, height);
    glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND);
    // No alpha blending — composite outputs fully opaque colour for fluid pixels
    // and SampleSky() for background pixels. Both are alpha=1 so blending is moot.

    glUseProgram(compositeProg);

    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, blurTexB);
    glUniform1i(u.comp_uCompTex, 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, normalTex);
    glUniform1i(u.comp_uNormalTex, 1);

    glUniform3fv(u.comp_uDirToSun, 1, glm::value_ptr(lightDirWorld));
    glUniformMatrix3fv(u.comp_uViewRotInv, 1, GL_FALSE, glm::value_ptr(viewRotInv));
    glUniform3fv(u.comp_uCamPosWorld, 1, glm::value_ptr(camPosWorld));

    glUniform3f(u.comp_uExtinction,
        settings.extinctionR, settings.extinctionG, settings.extinctionB);
    glUniform3f(u.comp_uShallowColor,
        settings.shallowColorR, settings.shallowColorG, settings.shallowColorB);
    glUniform3f(u.comp_uDeepColor,
        settings.deepColorR, settings.deepColorG, settings.deepColorB);

    glUniform3f(u.comp_uSkyZenith,
        settings.skyZenithR, settings.skyZenithG, settings.skyZenithB);
    glUniform3f(u.comp_uSkyHorizon,
        settings.skyHorizonR, settings.skyHorizonG, settings.skyHorizonB);
    glUniform3f(u.comp_uSkyGround,
        settings.skyGroundR, settings.skyGroundG, settings.skyGroundB);
    glUniform1f(u.comp_uSunIntensity, settings.sunIntensity);
    glUniform1f(u.comp_uSunInvSize, settings.sunInvSize);
    glUniform1f(u.comp_uReflStrength, settings.reflStrength);

    glUniform3f(u.comp_uBoundsSize,
        settings.boundsSizeX, settings.boundsSizeY, settings.boundsSizeZ);
    glUniform1f(u.comp_uTanHalfFov, tanHalfFov);
    glUniform1f(u.comp_uAspect, aspect);

    glBindVertexArray(quadVAO); glDrawArrays(GL_TRIANGLES, 0, 6); glBindVertexArray(0);
    glUseProgram(0);
}

// ─────────────────────────────────────────────────────────────────────────────
inline bool FluidRenderer::render(GLuint particleVAO, int particleCount,
    const glm::mat4& proj, const glm::mat4& view,
    int shaderType,
    const glm::vec3& lightDirWorld,
    float fovDegrees, float aspect) {
    // Legacy particles: return false so drawAll() uses the original program
    if (shaderType == 1) {
        settings.heateffect = true;
        return false;
    }
    if (!ready || particleCount <= 0) return false;

    // Heat colouring is invisible under the continuous SSF surface
    settings.heateffect = false;

    float tanHalfFov = tanf(glm::radians(fovDegrees * 0.5f));

    // View rotation inverse: transpose(mat3(view)) — orthonormal, so T = inverse.
    // Transforms vectors from view space to world space (rotation only, no translation).
    glm::mat3 viewRotInv = glm::transpose(glm::mat3(view));

    // Camera world position: view = [R | t] where t = -R * camPos → camPos = R^T * (-t)
    // view[3] is the 4th column (translation) in glm column-major storage.
    glm::vec3 camPosWorld = viewRotInv * (-glm::vec3(view[3]));

    passDepth(particleVAO, particleCount, proj, view);
    passThickness(particleVAO, particleCount, proj, view);
    passPack();
    passBlur();
    passNormal(viewRotInv, tanHalfFov, aspect);
    passComposite(glm::normalize(lightDirWorld), viewRotInv, camPosWorld, tanHalfFov, aspect);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
inline void FluidRenderer::cleanup() {
    destroyBuffers();
    if (quadVAO) { glDeleteVertexArrays(1, &quadVAO); quadVAO = 0; }
    if (quadVBO) { glDeleteBuffers(1, &quadVBO);       quadVBO = 0; }
    GLuint progs[] = { depthProg, thickProg, packProg, blurProg, normalProg, compositeProg };
    for (GLuint p : progs) if (p) glDeleteProgram(p);
    depthProg = thickProg = packProg = blurProg = normalProg = compositeProg = 0;
    ready = false;
}