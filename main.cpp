#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "fluid_sim/fluid_renderer.h"
#include <glm/gtc/type_ptr.hpp>
#include"fluid_sim/ui.h"
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>
#include <functional>
#include <chrono>
#include <omp.h>
#include <unordered_map>
#include "struct.h"
#include"fluid_sim/settings.h"
#include"source/compute.h"
#include "fluid_sim/main.h"

#include "fluid_sim/buttons.h"
#define _USE_MATH_DEFINES


//param settings;
//dt

FluidRenderer fluidRenderer;

const unsigned int screenWidth = 1800;
const unsigned int screenHeight = 900;

int currentWidth = (int)screenWidth;
int currentHeight = (int)screenHeight;

static double fuc_ms_avg = 0.0;
static int fuc_samples = 0;

// CHANGE: Replaced sf::View with custom View struct
struct View {
    float cx = screenWidth * 0.5f;
    float cy = screenHeight * 0.5f;
    float height = (float)screenHeight;
    float aspect = (float)screenWidth / (float)screenHeight;
    float zoom = 1.0f;
    float width() const { return height * aspect; }
} view;
















struct Camera {
    glm::vec3 position = glm::vec3(0.0f, -200.0f, 21.0f); // cinematic 3D
    glm::vec3 forward;
    glm::vec3 right;
    glm::vec3 up;

    float yaw = 90.0f;   // diagonal
    float pitch = -15.0f;  // looking down
    float fov = 70.0f;
};

Camera camera;
bool firstMouse = true;
double lastMouseX = 0, lastMouseY = 0;
bool cameraRotating = false;

float mouseSensitivity = 0.15f;
float scrollSensitivity = 2.0f;

bool mouseMassActive = false;
int mouseMassIndex = -1;   // index in bodies vector

// CHANGE: OpenGL vertex structure for batched rendering
struct GLVertex {
    float px, py, pz;           // world position
    float radius;           // radius in screen pixels
    float cr, cg, cb, ca;   // color
    float ox, oy;
    float wx, xy, xz;
    
};

// CHANGE: OpenGL resources
GLuint vao = 0, vbo = 0, ibo = 0;
size_t vbo_capacity = 0;
size_t ibo_capacity = 0;
GLuint program = 0;
GLuint bboxProgram = 0;
GLuint bboxVAO = 0;
GLuint bboxVBO = 0;

GLint loc_uProj = -1, loc_uView = -1;
GLint loc_uLightDir = -1, loc_uCameraPos = -1;
GLint bloc_uProj = -1, bloc_uView = -1, bloc_uColor = -1;

const char* boxvert = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 uProj;
uniform mat4 uView;

void main() {
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}

)glsl";
const char* boxfrag = R"glsl(
#version 330 core
out vec4 FragColor;
uniform vec3 uColor;

void main() {
    FragColor = vec4(uColor, 1.1);
}


)glsl";

const char* vertexShaderSource = R"glsl(
#version 330 core

layout(location = 0) in vec3 inCenterWorld;
layout(location = 1) in float inRadius;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec2 inOffset;
layout(location = 4) in float inHeat;   // b.heat ∈ [0,100]

uniform mat4 uProj;
uniform mat4 uView;
uniform vec3 uCameraPos;

out vec4 vColor;
out vec2 vOffset;
out float vHeat;
out vec3 vworld;
out vec3 vpos;
void main() {
    vec3 right = vec3(uView[0][0], uView[1][0], uView[2][0]);
    vec3 up    = vec3(uView[0][1], uView[1][1], uView[2][1]);

    vec3 worldPos =
        inCenterWorld +
        (right * inOffset.x + up * inOffset.y) * inRadius;

    gl_Position = uProj * uView * vec4(worldPos, 1.0);

    vColor = inColor;
    vOffset = inOffset; // ✅ FIXED
 vHeat   = inHeat;
vworld = uCameraPos;
vpos=worldPos;
}


)glsl";


const char* fragmentShaderSource = R"glsl(
#version 330 core

in vec2 vOffset;
in vec4 vColor;
in float vHeat;
in vec3 vworld;
in vec3 vpos;
out vec4 FragColor;

// direction TOWARDS the light (normalized)
uniform vec3 uLightDir;

void main() {
    // ---- circle cutout ----
    float r2 = dot(vOffset, vOffset);
    if (r2 > 1.0) discard;

    // ---- fake sphere normal ----
    float z = sqrt(1.0 - r2);
    vec3 normal = normalize(vec3(vOffset, z));

    // ---- lighting ----
   float diff = max(dot(normal, uLightDir), 0.20);
   float light =   0.3 + 0.7 * diff ;
float depth=length(vpos -vworld);
float depthFade = 1.0 - exp(-depth / 150.0);
    vec3 baseColor = vColor.rgb * light  ;

float blur= 1.0f;
if( r2>0.1){
blur=1.0 - (r2 -0.1);}
if(blur <=0)discard;

  
   
FragColor = vec4(baseColor  ,1.0f);
}



)glsl";

// CHANGE: Shader compilation helper
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetShaderInfoLog(s, 1024, nullptr, buf);
        std::cerr << "Shader compile error: " << buf << "\n";
    }
    return s;
}

static GLuint createProgram(const char* vs, const char* fs) {
    GLuint a = compileShader(GL_VERTEX_SHADER, vs);
    GLuint b = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, a);
    glAttachShader(p, b);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetProgramInfoLog(p, 1024, nullptr, buf);
        std::cerr << "Program link error: " << buf << "\n";
    }
    glDeleteShader(a); glDeleteShader(b);
    return p;
}

bool ensureVBOCapacity(size_t verts) {
    if (verts <= vbo_capacity) return false ;

    vbo_capacity = verts * 2 + 256;
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao == 0) glGenVertexArrays(1, &vao);

    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vbo_capacity * sizeof(GLVertex), nullptr, GL_STREAM_DRAW);

    GLsizei stride = sizeof(GLVertex);

    // Attribute 0: world position (px, py)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
        (void*)offsetof(GLVertex, px));
    // Attribute 1: radius (removed screen center, now just radius)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(GLVertex, radius));

    // Attribute 2: color (cr, cg, cb, ca)
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(GLVertex, cr));

    // Attribute 3: offset (ox, oy)
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(GLVertex, ox));


    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return true;
}

// CHANGE: IBO for indexed quad rendering
void ensureIBOCapacity(size_t numBodies) {
    size_t numIndices = numBodies * 6;
    if (numIndices <= ibo_capacity) return;

    ibo_capacity = numIndices * 2;
    if (ibo) glDeleteBuffers(1, &ibo);

    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, ibo_capacity * sizeof(GLuint), nullptr, GL_STATIC_DRAW);

    std::vector<GLuint> indices(ibo_capacity);
    for (size_t i = 0; i < ibo_capacity / 6; i++) {
        GLuint base = i * 4;
        indices[i * 6 + 0] = base + 0;
        indices[i * 6 + 1] = base + 1;
        indices[i * 6 + 2] = base + 2;
        indices[i * 6 + 3] = base + 0;
        indices[i * 6 + 4] = base + 2;
        indices[i * 6 + 5] = base + 3;
    }
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, indices.size() * sizeof(GLuint), indices.data());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}


// CHANGE: Orthographic projection matrix
void setOrtho(float left, float right, float bottom, float top, float nearv, float farv, float* out4x4) {
    float dx = right - left;
    float dy = top - bottom;
    float dz = farv - nearv;
    for (int i = 0; i < 16; ++i) out4x4[i] = 0.0f;
    out4x4[0] = 2.0f / dx;
    out4x4[5] = 2.0f / dy;
    out4x4[10] = -2.0f / dz;
    out4x4[12] = -(right + left) / dx;
    out4x4[13] = -(top + bottom) / dy;
    out4x4[14] = -(farv + nearv) / dz;
    out4x4[15] = 1.0f;
}

// CHANGE: World to screen coordinate conversion
inline void worldToScreen_topLeft(float wx, float wy, float& sx, float& sy, const View& v) {
    float halfH = v.height * 0.5f * v.zoom;
    float halfW = v.width() * 0.5f * v.zoom;
    float left = v.cx - halfW;
    float top = v.cy - halfH;
    float scale = (float)currentHeight / (v.height * v.zoom);
    sx = (wx - left) * scale;
    sy = (wy - top) * scale;
}


float y, p;


void restartSimulation() {
    
    settings.count = settings.totalBodies;
    freeDynamicGrid();
    settings.samplecount = 0;
    freegpu();
    initgpu(settings.maxparticles);
	initDynamicGrid(settings.maxparticles);
    registerBodies();
    settings.nopause = false;
  
    

}

void updatePhysics(float dt) {
    if (settings.fuc_ms > settings.maxframetime )settings.addParticle = false;
    
   
        computephysics(dt);
    
   
}
void initBoundingBox() {

    if (bboxVAO) { glDeleteVertexArrays(1, &bboxVAO); bboxVAO = 0; }
    if (bboxVBO) { glDeleteBuffers(1, &bboxVBO); bboxVBO = 0; }

    
    float boxVerts[] = {
        // bottom rectangle
        settings.minX, settings.minY, settings.minZ,   settings.maxX, settings.minY, settings.minZ,
        settings.maxX, settings.minY, settings.minZ,   settings.maxX, settings.maxY, settings.minZ,
        settings.maxX, settings.maxY, settings.minZ,   settings.minX, settings.maxY, settings.minZ,
        settings.minX, settings.maxY, settings.minZ,   settings.minX, settings.minY, settings.minZ,

        // top rectangle
        settings.minX, settings.minY, settings.maxz,   settings.maxX, settings.minY, settings.maxz,
        settings.maxX, settings.minY, settings.maxz,   settings.maxX, settings.maxY, settings.maxz,
        settings.maxX, settings.maxY, settings.maxz,   settings.minX, settings.maxY, settings.maxz,
        settings.minX, settings.maxY, settings.maxz,   settings.minX, settings.minY, settings.maxz,

        // vertical edges
        settings.minX, settings.minY, settings.minZ,   settings.minX, settings.minY, settings.maxz,
        settings.maxX, settings.minY, settings.minZ,   settings.maxX, settings.minY, settings.maxz,
        settings.maxX, settings.maxY, settings.minZ,   settings.maxX, settings.maxY, settings.maxz,
        settings.minX, settings.maxY, settings.minZ,   settings.minX, settings.maxY, settings.maxz,

        //mini spawn box verts
        settings.nx, settings.ny, settings.nz,   settings.mx, settings.ny, settings.nz,
        settings.mx, settings.ny, settings.nz,   settings.mx, settings.my, settings.nz,
        settings.mx, settings.my, settings.nz,   settings.nx, settings.my, settings.nz,
        settings.nx, settings.my, settings.nz,   settings.nx, settings.ny, settings.nz,

        settings.nx, settings.ny, settings.mz,   settings.mx, settings.ny, settings.mz,
        settings.mx, settings.ny, settings.mz,   settings.mx, settings.my, settings.mz,
        settings.mx, settings.my, settings.mz,   settings.nx, settings.my, settings.mz,
        settings.nx, settings.my, settings.mz,   settings.nx, settings.ny, settings.mz,

        settings.nx, settings.ny, settings.nz,   settings.nx, settings.ny, settings.mz,
        settings.mx, settings.ny, settings.nz,   settings.mx, settings.ny, settings.mz,
        settings.mx, settings.my, settings.nz,   settings.mx, settings.my, settings.mz,
        settings.nx, settings.my, settings.nz,   settings.nx, settings.my, settings.mz

		
    };

    glGenVertexArrays(1, &bboxVAO);
    glGenBuffers(1, &bboxVBO);

    glBindVertexArray(bboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, bboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(boxVerts), boxVerts, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindVertexArray(0);
}
void calcKernels() {
    float h2 = settings.h * settings.h;
    float h3 = settings.h * settings.h * settings.h;
    float h4 = h2 * h2;
	float h5 = h2 * h3;
    float h6 = h3 * h3;
    float h9 = h3 * h3 * h3;

    
   
   
        
     //  settings.rest_density = 0.1036f * powf(3.5f / settings.h, 3.0f);



    settings.pollycoef6 = 315.0f / (64.0f * settings.pi * h9);
    //settings.spikycoef2 = 15.0f / (settings.pi * h5);
   settings.Sdensity = settings.pollycoef6 * h6;//self density at r=0
   settings.spikycoef = 15.0f / (settings.pi * h6);
   settings.ndensity = settings.spikycoef * h3;//near self density at r=0
   settings.spikygradv = -45 / (settings.pi * h6);
   settings.viscosity = 45 / (settings.pi * h6);
   settings.h2 = settings.h * settings.h;
}


void drawAll() {

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // Ensure VBO is large enough (first frame, or after MaxFps growth).
   // If it had to grow, re-register it with CUDA.
    if (ensureVBOCapacity((size_t)settings.count * 3)) {
        unregisterGLBuffer();
        registerGLBuffer(vbo);
    }

    

  

    glm::mat4 proj = glm::perspective(
        glm::radians(camera.fov),
        (float)currentWidth / (float)currentHeight,
        0.1f,
        20000.0f
    );

    glm::mat4 viewMat = glm::lookAt(
        camera.position,
        camera.position + camera.forward,
        camera.up
    );
    glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 0.6f, 1.0f));
    float aspect = (float)currentWidth / (float)screenHeight;

 
    if (settings.boundingBox) {
        // ── bounding box pass ────────────────────────────────────────────────────
        glUseProgram(bboxProgram);
        glUniformMatrix4fv(bloc_uProj, 1, GL_FALSE, glm::value_ptr(proj));
        glUniformMatrix4fv(bloc_uView, 1, GL_FALSE, glm::value_ptr(viewMat));
        glUniform3f(bloc_uColor, 1.0f, 1.0f, 1.0f);

        glBindVertexArray(bboxVAO);
        glLineWidth(1.0f);
        if (settings.nopause) {

        glDrawArrays(GL_LINES, 0, 24);
        }
        else {
            glDrawArrays(GL_LINES, 0, 48);
        }

        glBindVertexArray(0);
        glUseProgram(0);
    }
    



    
      bool rendered=  fluidRenderer.render(vao, settings.count,
            proj, viewMat,
            settings.shaderType,
            lightDir,
            camera.fov, aspect);
    
    

      if (!rendered) {
          glUseProgram(program);
          glUniformMatrix4fv(loc_uProj, 1, GL_FALSE, glm::value_ptr(proj));
          glUniformMatrix4fv(loc_uView, 1, GL_FALSE, glm::value_ptr(viewMat));
          glUniform3fv(loc_uLightDir, 1, glm::value_ptr(lightDir));
          glUniform3f(loc_uCameraPos, settings.wx, settings.wy, settings.wz);

          glBindVertexArray(vao);
          glDrawArrays(GL_TRIANGLES, 0, settings.count * 3);
          glBindVertexArray(0);
          glUseProgram(0);
      }
	
  
	
}

void updateCameraVectors(Camera& cam)
{
    float yawRad = glm::radians(cam.yaw);
    float pitchRad = glm::radians(cam.pitch);

    // Z-up world
    cam.forward = glm::normalize(glm::vec3(
        cos(yawRad) * cos(pitchRad),
        sin(yawRad) * cos(pitchRad),
        sin(pitchRad)
    ));

    cam.right = glm::normalize(glm::cross(cam.forward, glm::vec3(0, 0, 1)));
    cam.up = glm::normalize(glm::cross(cam.right, cam.forward));
}
void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    // Only rotate when holding left click
    if (!cameraRotating) {
        lastMouseX = xpos;
        lastMouseY = ypos;
        return;
    }

    if (firstMouse) {
        lastMouseX = xpos;
        lastMouseY = ypos;
        firstMouse = false;
    }

    float dx = (float)(xpos - lastMouseX);
    float dy = (float)(lastMouseY - ypos);

    lastMouseX = xpos;
    lastMouseY = ypos;

    dx *= mouseSensitivity;
    dy *= mouseSensitivity;

    camera.yaw -= dx;
    camera.pitch += dy;

    if (camera.pitch > 89.0f)  camera.pitch = 89.0f;
    if (camera.pitch < -89.0f) camera.pitch = -89.0f;

    updateCameraVectors(camera);
}
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            cameraRotating = true;

            // 🔴 HARD RESET mouse origin
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
            firstMouse = false;
        }
        else if (action == GLFW_RELEASE) {
            cameraRotating = false;
            firstMouse = true; // prepare for next drag
        }
    }
}
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    camera.fov -= (float)yoffset * scrollSensitivity;

    if (camera.fov < 15.0f)  camera.fov = 15.0f;
    if (camera.fov > 120.0f) camera.fov = 120.0f;
}
void updateCameraMovement(GLFWwindow* window, float dt) {

    float speed = 250.0f * dt;  // tweak this

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.position += camera.forward * speed;

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.position -= camera.forward * speed;

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.position -= camera.right * speed;

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.position += camera.right * speed;

    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        camera.position += camera.up * speed;

    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        camera.position -= camera.up * speed;


}

void framebuffer_size_callback(GLFWwindow* w, int width, int height) {
    if (width == 0 || height == 0) return;
    currentWidth = width;
    currentHeight = height;
    glViewport(0, 0, width, height);
    view.cx = width * 0.5f;
    view.cy = height * 0.5f;
    view.height = (float)height;
    view.aspect = (float)width / (float)height;
}
int main() {
    srand((unsigned)time(nullptr));
    // CHANGE: Initialize GLFW instead of SFML
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n"; return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "fluid Simulation - OpenGL", nullptr, nullptr);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    if (!window) { std::cerr << "Failed create window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    // CHANGE: Initialize GLAD instead of SFML GL
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n"; return -1;
    }
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
   // glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    glViewport(0, 0, screenWidth, screenHeight);

    fluidRenderer.init(screenWidth, screenHeight);
    // CHANGE: Setup callbacks before ImGui
    updateCameraVectors(camera);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    glfwSetScrollCallback(window, scrollCallback);

    // CHANGE: Initialize ImGui for GLFW+OpenGL3
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    const char* glsl_version = "#version 330";
    ImGui_ImplOpenGL3_Init(glsl_version);

    // CHANGE: Create shader program
    program = createProgram(vertexShaderSource, fragmentShaderSource);
    bboxProgram = createProgram(boxvert, boxfrag);

    loc_uProj = glGetUniformLocation(program, "uProj");
    loc_uView = glGetUniformLocation(program, "uView");
    loc_uLightDir = glGetUniformLocation(program, "uLightDir");
    loc_uCameraPos = glGetUniformLocation(program, "uCameraPos");
    bloc_uProj = glGetUniformLocation(bboxProgram, "uProj");
    bloc_uView = glGetUniformLocation(bboxProgram, "uView");
    bloc_uColor = glGetUniformLocation(bboxProgram, "uColor");



    calcKernels();
    initBoundingBox();
    initgpu(settings.maxparticles);
    initDynamicGrid(settings.maxparticles);
    
    ensureVBOCapacity((size_t)500000*3);
    registerGLBuffer(vbo);

    registerBodies();
    
   
	restartSimulation();
    const float targetFPS = 60.0f;
    const float upperThreshold = 65.0f;
    const float lowerThreshold = 55.0f;
   
   
    float debugtime = 0.0f;

   
    
    double lastTime = glfwGetTime();
    double fpsClock = lastTime;

    view.cx = screenWidth * 0.5f;
    view.cy = screenHeight * 0.5f;
    view.height = (float)screenHeight;
    view.aspect = (float)screenWidth / (float)screenHeight;
    view.zoom = 1.0f;

    while (!glfwWindowShouldClose(window)) {
       
        glfwPollEvents();

        ui_init();
        if (settings.count >= (settings.maxparticles)*0.98f) {
            settings.addParticle = false;
	   }

        // Timing
        double now = glfwGetTime();
        double frameTime = now - lastTime;
        lastTime = now;
        settings.accumulator += (float)frameTime;
        float dt = (float)frameTime;
        
        updateCameraMovement(window, dt);
        buttons(window);
        y = camera.yaw;
        p = camera.pitch;
        settings.wx = camera.position.x;
        settings.wy = camera.position.y;
        settings.wz = camera.position.z;
        
        

        if (settings.recordSim) {
           
                updatePhysics(settings.fixedDt);
            settings.accumulator = 0.0f;
        }
        else {

            float effectiveDt = settings.fixedDt * settings.simspeed;
            while (settings.accumulator >= settings.fixedDt) {


              
                    updatePhysics(effectiveDt);
                



                settings.accumulator -= settings.fixedDt;
            }
        }

       
        /*if (debugtime > 0.50f) {
            printf("ms %3f\n", sample_ms);
            debugtime = 0.0f;
        }
        debugtime += effectiveDt;*/
       
        // CHANGE: OpenGL rendering instead of SFML
        glClearColor(settings.bgColorR, settings.bgColorG, settings.bgColorB, 1.0f);
      



        drawAll();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        // FPS measurement
        double elapsed = now - fpsClock;
        fpsClock = now;
        settings.fps = (elapsed > 0.0) ? 1.0 / elapsed : settings.fps;
        settings.fpsTimer += (float)elapsed;
        settings.fpsCount++;
        if (settings.fps > settings.maxFps) settings.maxFps = (float)settings.fps;
        if (settings.fps < settings.minFps) settings.minFps = (float)settings.fps;
        if (settings.fpsTimer >= 0.5f) {
            settings.avgFps = settings.fpsCount / settings.fpsTimer;
            settings.fpsTimer = 0.f;
            settings.fpsCount = 0;
        }
       settings.fuc_ms= (settings.avgFps > 0.0f) ? 1000.0f / settings.avgFps : 0.0f;
    }
    printf("bboxVAO=%u bboxVBO=%u bboxProgram=%u\n",
        bboxVAO, bboxVBO, bboxProgram);

    unregisterGLBuffer();
    // CHANGE: Cleanup OpenGL resources
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (program) glDeleteProgram(program);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ibo) glDeleteBuffers(1, &ibo);
    if (vao) glDeleteVertexArrays(1, &vao);
    fluidRenderer.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();


    return 0;
}