#pragma once

#include <glad/glad.h>
#include "settings.h"

extern "C" {
    extern GLuint floorProgram;
    extern GLuint floorVAO;
    extern GLuint floorVBO;
    extern GLint uTileSize;
    extern GLint uFloorSize;
    extern GLint uVariation;
    extern GLint uFloorCenterX;
    extern GLint uFloorCenterY;
    extern GLint floor_uProj;
    extern GLint floor_uView;
    extern GLint floor_uLightDir;
    extern GLint floor_uCameraPos;
    extern GLint uColor1;
    extern GLint uColor2;
    extern GLint uColor3;
    extern GLint uColor4;

    extern const char* floorVertexShader;
    extern const char* floorFragmentShader;

    extern float floorverts[];
    extern const size_t floorverts_count;
    void initFloor();
}


