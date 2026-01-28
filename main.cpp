#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


#include <glm/gtc/type_ptr.hpp>

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
#include"source/compute.h"
#define _USE_MATH_DEFINES



//dt

float fixedDt = 1 / 120.0f;


const unsigned int screenWidth = 1800;
const unsigned int screenHeight = 900;

static double fuc_ms_avg = 0.0;
static int fuc_samples = 0;
static double fuc_ms = 0.0;
// CHANGE: Replaced sf::View with custom View struct
struct View {
    float cx = screenWidth * 0.5f;
    float cy = screenHeight * 0.5f;
    float height = (float)screenHeight;
    float aspect = (float)screenWidth / (float)screenHeight;
    float zoom = 1.0f;
    float width() const { return height * aspect; }
} view;











const float M_PI = 3.14159265359f;

constexpr float MAX_HEAT = 100.0f;
constexpr float HEAT_TO_COLOR = 2.0f;


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
}


)glsl";

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
const char* fragmentShaderSource = R"glsl(
#version 330 core

in vec2 vOffset;
in vec4 vColor;
in float vHeat;
in vec3 vworld;
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
   float light =   1.1 + 0.9 * diff ;

    vec3 baseColor = vColor.rgb  * light  ;


  
   /*// edge softness (screen-space, anti-aliased)
    float edge = 1.0 - smoothstep(
        0.9,
        1.0,
        r2 + fwidth(r2) * 5.0
    );

    // heat-controlled blur strength
    float blur = clamp(vHeat, 0.0, 1.0);

    // glow color (slightly hotter tint)
    vec3 glowColor = baseColor * (1.2 + blur);

    // mix sharp core with soft edge
    vec3 finalColor = mix(glowColor, baseColor, edge);

    // alpha fades softly at edges
    float alpha = edge;

    FragColor = vec4(finalColor, alpha);*/
float blur= 1.0f;
if( r2>0.1){
blur=1.0 - (r2 -0.1);}
if(blur <=0){
blur =0.1;}
FragColor = vec4(baseColor  ,blur);
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

void ensureVBOCapacity(size_t verts) {
    if (verts <= vbo_capacity) return;

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
    float scale = (float)screenHeight / (v.height * v.zoom);
    sx = (wx - left) * scale;
    sy = (wy - top) * scale;
}




//ui////////////
//functions
bool colisionFun = true;

bool updateFun = true;

//simspeed
float simspeed = 1.0f;




//particle settings

int totalBodies = 10000;
float size = 1.5f;
float mmass = 10.0f;




int substeps = 1;

//camera
float y, p;

//heat
float cold = 4.500f;
float cr = 4.0f;
float hmulti = 15.0f;


//world cords
float wx, wy, wz;


//sph variables
float h = 3.5f;
float cellsize = h*1.5f;
float h2 = h * h;
float rest_density = 100.0f;//density-idk
float pressure = 200.0f;//pressure-idk--K

float visc = 0.01f;
bool heateffect = true;

float downf = 2.0f;


bool addparticle = false;
//
 
std::vector<float> posx, posy, posz;
std::vector<float> aclx, acly, aclz;
std::vector<float> old_aclx, old_acly, old_aclz;
std::vector<float> velx, vely, velz;
std::vector<float> forcex, forcey, forcez;
std::vector<float> Size;
std::vector<float> Mass;
std::vector<int> Iscenter;
std::vector<int> r, g, b;
std::vector<int> br, bg, bb;
std::vector<float> Heat;
std::vector<float> Density;
std::vector<float> Pressure;

float minX = -50.0f, maxX = 50.0f;
float minY = -50.0f, maxY = 50.0f;
float restitution = 0.8f;
float minZ = 0.0f;
float maxz = 100.0f;
float as = 1.0f;
float st = 1.0f;
//////////////////////////////////////
//register particles
void registerBody() {
  

    for (int i = 0; i < totalBodies; ++i) {
        
        float m = mmass;
		float s = size;
        float is = 0;

        float x = 0, y = 0, z = 0;
        float vx = 0, vy = 0, vz = 0;
        float particle_spacing = h*0.6f;

        int particles_per_side = (int)ceil(cbrt((float)totalBodies));
        int maxXcount = (int)((maxX - minX) / particle_spacing);
        int maxYcount = (int)((maxY - minY) / particle_spacing);
        int maxZcount = (int)((maxz - minZ) / particle_spacing);

        // Use cubic grid but respect physical limits
        int nx = std::min(particles_per_side, std::max(1, maxXcount));
        int ny = std::min(particles_per_side, std::max(1, maxYcount));
        int nz = std::min(particles_per_side, std::max(1, maxZcount));

        // Convert flattened index to 3D grid coordinates
        int ix = i % nx;
        int iy = (i / nx) % ny;
        int iz = i / (nx * ny);

        // Calculate grid dimensions
        float gridSizeX = (nx - 1) * particle_spacing;
        float gridSizeY = (ny - 1) * particle_spacing;
        float gridSizeZ = (nz - 1) * particle_spacing;

        // Calculate starting position with half particle spacing offset from edges
        float startX = (minX) + particle_spacing * maxX/2;
        float startY = (minY) + particle_spacing * maxY/2;
        float startZ = maxz - particle_spacing * 1.5f;  // Offset from top

        // Generate grid
        x = startX + ix * particle_spacing ;
        y = startY + iy * particle_spacing ;
        z = startZ - iz * particle_spacing;  // Start at maxZ with offset, go downward

        float d = rest_density;   // NOT 0
        float p = 0.0f;
       
       
        int Br = 255;
        int Bg = 255;
        int Bb = 255;

		posx.push_back(x);
        posy.push_back(y);
        posz.push_back(z);
        velx.push_back(vx);
        vely.push_back(vy);
        velz.push_back(vz);
        aclx.push_back(0.0f);
        acly.push_back(0.0f);
        aclz.push_back(0.0f);
        old_aclx.push_back(0.0f);
        old_acly.push_back(0.0f);
        old_aclz.push_back(0.0f);
        forcex.push_back(0.0f);
        forcey.push_back(0.0f);
        forcez.push_back(0.0f);
        Size.push_back(s);
        Mass.push_back(m);
        Iscenter.push_back(is);
        br.push_back(Br);
        bg.push_back(Bg);
        bb.push_back(Bb);
        r.push_back(br[i]);
        g.push_back(bg[i]);
        b.push_back(bb[i]);
        Heat.push_back(0.0f);
        Density.push_back(d);
		Pressure.push_back(p);
    }

}
inline float randf(float min, float max) {
    return min + (max - min) * (float(rand()) / float(RAND_MAX));
}

void restartSimulation() {
	posx.clear();
	posy.clear();
	posz.clear();
	velx.clear();
	vely.clear();
    velz.clear();
	aclx.clear();
	acly.clear();
	aclz.clear();
	old_aclx.clear();
	old_acly.clear();
	old_aclz.clear();
	forcex.clear();
	forcey.clear();
	forcez.clear();
	Size.clear();
	Mass.clear();
	Iscenter.clear();
	r.clear();
	g.clear();
	b.clear();
	br.clear();
	bg.clear();
	bb.clear();
	Heat.clear();
	Density.clear();
	Pressure.clear();
  
    freeDynamicGrid();
    freegpu();
    registerBody();
	initgpu(posx.size());
	initDynamicGrid(posx.size());
  
    copyarray(posx.size(),
        posx.data(),
        posy.data(),
        posz.data(),
        velx.data(),
        vely.data(),
        velz.data(),
        aclx.data(),
        acly.data(),
        aclz.data(),
        old_aclx.data(),
        old_acly.data(),
        old_aclz.data(),
        forcex.data(),
        forcey.data(),
        forcez.data(),
        Size.data(),
        Mass.data(),
        Iscenter.data(),
        r.data(),
        g.data(),
        b.data(),
        br.data(),
        bg.data(),
        bb.data(),
        Heat.data(),
        Density.data(),
        Pressure.data()



    );

    }

int rc = 255;
int gc = 255;
int bc = 255;
void updatePhysics(float dt) {
    float subDt = dt / (float)substeps;


    for (int step = 0; step < substeps; step++) {
        
        if (addparticle == true) {
           // addparticles(totalBodies,maxz,size,mmass);
        }
        

      

        if (colisionFun == true) {


            stepsph(posx.size(), subDt, h, pressure, rest_density,minX,minY,minZ,maxX,maxY,maxz,visc);

        }
        if (heateffect == true) {
            heating(posx.size(), subDt, hmulti, cold,rc,gc,bc);
        }



        if (updateFun == true) {
			
            updatebodies(subDt, posx.size(), cold, MAX_HEAT,minX,maxX,minY,maxY,minZ,maxz,restitution, downf

            );
        }

      


    }
    updatearray(posx.size(),
        posx.data(),
        posy.data(),
        posz.data(),
        Size.data(),
        r.data(),
        g.data(),
        b.data());
       

    
}
void initBoundingBox() {
    
    float boxVerts[] = {
        // bottom rectangle
        minX, minY, minZ,   maxX, minY, minZ,
        maxX, minY, minZ,   maxX, maxY, minZ,
        maxX, maxY, minZ,   minX, maxY, minZ,
        minX, maxY, minZ,   minX, minY, minZ,

        // top rectangle
        minX, minY, maxz,   maxX, minY, maxz,
        maxX, minY, maxz,   maxX, maxY, maxz,
        maxX, maxY, maxz,   minX, maxY, maxz,
        minX, maxY, maxz,   minX, minY, maxz,

        // vertical edges
        minX, minY, minZ,   minX, minY, maxz,
        maxX, minY, minZ,   maxX, minY, maxz,
        maxX, maxY, minZ,   maxX, maxY, maxz,
        minX, maxY, minZ,   minX, maxY, maxz
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

void drawAll() {

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND); // IMPORTANT: no transparency
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const int VERTS_PER_BODY = 3;
    size_t totalVerts = posx.size() * VERTS_PER_BODY;

    ensureVBOCapacity(totalVerts);
    ensureIBOCapacity(posx.size());

    std::vector<GLVertex> verts(totalVerts);

    for (int i = 0; i < posx.size(); i++) {
        

        float cr = r[i] / 255.0f;
        float cg = g[i] / 255.0f;
        float cb = b[i] / 255.0f;

        int v = i * 3;

        // Fullscreen-covering triangle offsets
        const float ox[3] = { -1.0f,  3.0f, -1.0f };
        const float oy[3] = { -1.0f, -1.0f,  3.0f };

        for (int k = 0; k < 3; k++) {
            verts[v + k].px = posx[i];
            verts[v + k].py = posy[i];
            verts[v + k].pz = posz[i];

            verts[v + k].radius = Size[i];

            verts[v + k].cr = cr;
            verts[v + k].cg = cg;
            verts[v + k].cb = cb;
            verts[v + k].ca = 1.10f;

            verts[v + k].ox = ox[k];
            verts[v + k].oy = oy[k];


        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        verts.size() * sizeof(GLVertex),
        verts.data()
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(program);

    glm::mat4 proj = glm::perspective(
        glm::radians(camera.fov),
        (float)screenWidth / (float)screenHeight,
        0.1f,
        20000.0f
    );

    glm::mat4 view = glm::lookAt(
        camera.position,
        camera.position + camera.forward,
        camera.up
    );

    glUniformMatrix4fv(
        glGetUniformLocation(program, "uProj"),
        1, GL_FALSE, glm::value_ptr(proj)
    );

    glUniformMatrix4fv(
        glGetUniformLocation(program, "uView"),
        1, GL_FALSE, glm::value_ptr(view)
    );
    glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 0.6f, 1.0f));

    glUniform3f(
        glGetUniformLocation(program, "uLightDir"),
        lightDir.x,
        lightDir.y,
        lightDir.z
    );
    glUniform3f(
        glGetUniformLocation(program, "uCameraPos"),
        wx,
        wy,
        wz
    );

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, posx.size() * 3);
    glBindVertexArray(0);

    glUseProgram(0);

    glUseProgram(bboxProgram); // or reuse program if compatible

    glUniformMatrix4fv(
        glGetUniformLocation(bboxProgram, "uProj"),
        1, GL_FALSE, glm::value_ptr(proj)
    );

    glUniformMatrix4fv(
        glGetUniformLocation(bboxProgram, "uView"),
        1, GL_FALSE, glm::value_ptr(view)
    );

    // white lines
    glUniform3f(
        glGetUniformLocation(bboxProgram, "uColor"),
        1.0f, 1.0f, 1.0f
    );
   // glDisable(GL_DEPTH_TEST);

    glBindVertexArray(bboxVAO);
    glLineWidth(1.0f);

    glDrawArrays(GL_LINES, 0, 24); // 12 edges * 2 verts
    glBindVertexArray(0);

   // glEnable(GL_DEPTH_TEST);
    glUseProgram(0);
}
bool option3 = false;
void MaxFps(double avgMs) {
    const float target = 16.67f;
    const int minBodies = 100;
    const int maxBodies = 500000;

    float error = avgMs - target;

    if (fabs(error) < 0.5f) return; // already stable

    float adjustFactor = 0.05f; // 5% per update

    if (error > 0.0f) {
        // too slow → reduce bodies
        totalBodies = (int)(totalBodies * (1.0f - adjustFactor));
    }
    else {
        // too fast → add bodies
        totalBodies = (int)(totalBodies * (1.0f + adjustFactor));
    }

    totalBodies = std::clamp(totalBodies, minBodies, maxBodies);

    restartSimulation();
   
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

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camera.position += camera.up * speed;

    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        camera.position -= camera.up * speed;


}
void buttons(GLFWwindow* window) {

    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        restartSimulation();
}
void framebuffer_size_callback(GLFWwindow* w, int width, int height) {
    glViewport(0, 0, width, height);
}
int main() {
    srand((unsigned)time(nullptr));
    // CHANGE: Initialize GLFW instead of SFML
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n"; return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);


    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "N-Body Simulation - OpenGL", nullptr, nullptr);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    if (!window) { std::cerr << "Failed create window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    // CHANGE: Initialize GLAD instead of SFML GL
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n"; return -1;
    }
    glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    glViewport(0, 0, screenWidth, screenHeight);

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
    ensureVBOCapacity(1024);



    float accumulator = 0.f;
    float fps = 0.f, avgFps = 0.f, maxFps = 0.f, minFps = 9999.f;
    float fpsTimer = 0.f;
    int fpsCount = 0;


    const float targetFPS = 60.0f;
    const float upperThreshold = 65.0f;
    const float lowerThreshold = 55.0f;
    initBoundingBox();
    registerBody();
    initgpu(posx.size());
    copyarray(posx.size(),
        posx.data(),
        posy.data(),
        posz.data(),
        velx.data(),
        vely.data(),
        velz.data(),
        aclx.data(),
        acly.data(),
        aclz.data(),
        old_aclx.data(),
        old_acly.data(),
        old_aclz.data(),
        forcex.data(),
        forcey.data(),
        forcez.data(),
        Size.data(),
        Mass.data(),
        Iscenter.data(),
        r.data(),
        g.data(),
        b.data(),
        br.data(),
        bg.data(),
        bb.data(),
        Heat.data(),
        Density.data(),
        Pressure.data()



    );

	initDynamicGrid(posx.size());
   
    
    double lastTime = glfwGetTime();
    double fpsClock = lastTime;

    view.cx = screenWidth * 0.5f;
    view.cy = screenHeight * 0.5f;
    view.height = (float)screenHeight;
    view.aspect = (float)screenWidth / (float)screenHeight;
    view.zoom = 1.0f;

    while (!glfwWindowShouldClose(window)) {

        glfwPollEvents();


        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // UI 
        ImGui::Begin("Settings");
        if (posx.size() != totalBodies) {
            ImGui::Text("body count error");
        }
        ImGui::Text("FPS: %.0f (Min: %.0f / Max: %.0f)", avgFps, minFps, maxFps);
       
        ImGui::Text("physics: %5.3f ms", fuc_ms);
        ImGui::Text("yaw: %.00f  pitch: %.00f", y, p);
        ImGui::Text("x:%.0f y %.0f z %.0f", wx, wy, wz);

        ImGui::Text("variables");
        ImGui::SliderFloat("speed", &simspeed, 0.001f, 10.0f);

        ImGui::SliderFloat("color fade speed", &cold,0.1f,20.0f);
        ImGui::SliderFloat("color gen speed", &hmulti,0.1f,20.0f);
      
       
        ImGui::SliderFloat("smoothing", &h, 0.0f, 20.0f);

        ImGui::InputFloat("rest density", &rest_density);
       // ImGui::SliderFloat("rest density", &rest_density,0.001f,100.0f);
        ImGui::InputFloat("pressure f", &pressure);
       // ImGui::SliderFloat("pressure f", &pressure,0.001f,1000.0f);
       // ImGui::SliderFloat("density multiplier", &densitymultiplier,0.001f,10.0f);
        
        ImGui::InputFloat("viscosity", &visc);
      

       
        ImGui::SliderFloat("G", &downf,0.0f,1000.0f);
        ImGui::InputFloat("res", &restitution);
    
        ImGui::SliderFloat("boundx", &maxX, 1.0f, 500.0f); {

            minX = -maxX;
            initBoundingBox();
        };
        ImGui::SliderFloat("boundy", &maxY,1.0f,500.0f);
        {

            minY = -maxY;
            initBoundingBox();
        }
      
        ImGui::Text("material settings");
        ImGui::InputInt("Total Bodies", &totalBodies);
        if (ImGui::IsItemDeactivatedAfterEdit()) {

            restartSimulation();
        }
        ImGui::SliderInt("color r", &rc,0,255);
        ImGui::SliderInt("color g", &gc,0,255);
        ImGui::SliderInt("color b", &bc,0,255);


        ImGui::InputFloat("size", &size);
        if (ImGui::IsItemDeactivatedAfterEdit()) {

            restartSimulation();
        }
        ImGui::InputFloat("mass", &mmass);
        if (ImGui::IsItemDeactivatedAfterEdit()) {

            restartSimulation();
        }
        ImGui::Text("physics");
        ImGui::Checkbox("sph", &colisionFun);
      
        ImGui::Checkbox("update bodies", &updateFun);
        ImGui::Checkbox("max at 60 fps", &option3);

        ImGui::Checkbox("add particles ", &addparticle);

        






        ImGui::Text("performance");




        ImGui::Text("physics: %5.3f ms", fuc_ms);
        ImGui::InputInt("substeps", &substeps);
      


       

        ImGui::End();
        // Timing
        double now = glfwGetTime();
        double frameTime = now - lastTime;
        lastTime = now;
        accumulator += (float)frameTime;
        float dt = (float)frameTime;
        
        updateCameraMovement(window, dt);
        buttons(window);
        y = camera.yaw;
        p = camera.pitch;
        wx = camera.position.x;
        wy = camera.position.y;
        wz = camera.position.z;
        
        float effectiveDt = fixedDt * simspeed;
        auto t0 = std::chrono::high_resolution_clock::now();
        while (accumulator >= fixedDt) {



            updatePhysics(effectiveDt);




            accumulator -= fixedDt;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        fuc_ms_avg += ms;
        fuc_samples++;
        if (fuc_samples >= 60) {   // 1 second @ 60 FPS
            fuc_ms = fuc_ms_avg / fuc_samples;
            fuc_ms_avg = 0.0;
            fuc_samples = 0;
        }
        if (option3) {
            MaxFps(fuc_ms);
        }
       
        // CHANGE: OpenGL rendering instead of SFML
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);



        drawAll();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        // FPS measurement
        double elapsed = glfwGetTime() - fpsClock;
        fpsClock = glfwGetTime();
        fps = (elapsed > 0.0) ? 1.0 / elapsed : fps;
        fpsTimer += (float)elapsed;
        fpsCount++;
        if (fps > maxFps) maxFps = (float)fps;
        if (fps < minFps) minFps = (float)fps;
        if (fpsTimer >= 0.5f) {
            avgFps = fpsCount / fpsTimer;
            fpsTimer = 0.f;
            fpsCount = 0;
        }
    }
    printf("bboxVAO=%u bboxVBO=%u bboxProgram=%u\n",
        bboxVAO, bboxVBO, bboxProgram);

    // CHANGE: Cleanup OpenGL resources
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (program) glDeleteProgram(program);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ibo) glDeleteBuffers(1, &ibo);
    if (vao) glDeleteVertexArrays(1, &vao);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}