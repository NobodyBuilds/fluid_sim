#include <iostream>
#include "floor.h"
#include "main.h"
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "settings.h"
#include <cstring>  // for memcpy

GLint floor_uCameraPos = -1;
GLint floor_uLightDir = -1;
GLint floor_uProj = -1;
GLint floor_uView = -1;
GLuint floorProgram = 0;
GLuint floorVAO = 0, floorVBO = 0;
GLint uTileSize = -1;
GLint uFloorSize = -1;
GLint uVariation = -1;
GLint uFloorCenterX = -1;
GLint uFloorCenterY = -1;

GLint uColor1 = -1;
GLint uColor2 = -1;
GLint uColor3 = -1;
GLint uColor4 = -1;

const char* floorVertexShader = R"glsl(
#version 330 core

layout(location = 0) in vec3 aPos;

uniform mat4 uProj;
uniform mat4 uView;

out vec3 vPos;

void main() {
    vPos = aPos;
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}
)glsl";

const char* floorFragmentShader = R"glsl(
#version 330 core

in vec3 vPos;
out vec4 FragColor;

uniform vec3 uLightDir;

// 🧠 NEW uniforms (from CPU)
uniform float uTileSize;
uniform float uFloorSize;
uniform float uVariation;
uniform float uFloorCenterX;
uniform float uFloorCenterY;

// 🎨 4 quadrant colors
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
uniform vec3 uColor4;

// --- hash function ---
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {

    // normalize position into floor space (0 → floorSize)
    vec2 pos = vPos.xy;

    // tile coords
    vec2 tile = floor(pos / uTileSize);

    float checker = mod(tile.x + tile.y, 2.0);

    // 🧠 quadrant selection (centered at floor center for equal quadrants)
    vec3 col;

    if (pos.x > uFloorCenterX && pos.y > uFloorCenterY)
        col = uColor1;
    else if (pos.x < uFloorCenterX && pos.y > uFloorCenterY)
        col = uColor2;
    else if (pos.x < uFloorCenterX && pos.y < uFloorCenterY)
        col = uColor3;
    else
        col = uColor4;

    // checker mix (slight variation between light/dark tiles)
    vec3 baseColor = mix(col * 1.1, col * 0.85, checker);

    // 🎲 variation
    float rnd = hash(tile);
    baseColor += (rnd - 0.5) * uVariation;

    // 🌗 lighting
    vec3 normal = vec3(0.0, 0.0, 1.0);
    float diff = max(dot(normal, normalize(uLightDir)), 0.2);

    vec3 finalColor = baseColor * (0.3 + 0.7 * diff);

    FragColor = vec4(finalColor, 1.0);
}


)glsl";

float floorverts[] = {
    // These will be updated dynamically in initFloor()
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f
};
const size_t floorverts_count = sizeof(floorverts) / sizeof(float);
extern "C" void initFloor() {

    if (floorVAO) { glDeleteVertexArrays(1, &floorVAO); floorVAO = 0; }
    if (floorVBO) { glDeleteBuffers(1, &floorVBO); floorVBO = 0; }

    // Calculate individual bounds from floorbounds
 

    // Update vertex data with current settings
    float verts[] = {
        // triangle 1
        settings.floorboun_x, settings.floorboun_y, settings.minZ-1.0f,
        settings.floorbounx, settings.floorboun_y, settings.minZ-1.0f,
        settings.floorbounx, settings.floorbouny, settings.minZ-1.0f,

        // triangle 2
        settings.floorbounx, settings.floorbouny, settings.minZ-1.0f,
        settings.floorboun_x, settings.floorbouny, settings.minZ-1.0f,
        settings.floorboun_x, settings.floorboun_y, settings.minZ-1.0f
    };

    // Copy to global array for compatibility
    memcpy(floorverts, verts, sizeof(verts));

    glGenVertexArrays(1, &floorVAO);
    glGenBuffers(1, &floorVBO);

    glBindVertexArray(floorVAO);
    glBindBuffer(GL_ARRAY_BUFFER, floorVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(floorverts), floorverts, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindVertexArray(0);
}