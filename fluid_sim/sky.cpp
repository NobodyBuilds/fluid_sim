#include<iostream>
#include<glad/glad.h>
#include<algorithm>
#include"sky.h"
#include <fstream>
#include <sstream>
#include <string>
#include <glm/gtc/type_ptr.hpp>

SkyRenderer sky;
         
GLuint loadShaderProgram(const char* vertPath, const char* fragPath) {
    auto readFile = [](const char* path) {
        std::ifstream f(path);
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
        };

    std::string vertSrc = readFile(vertPath);
    std::string fragSrc = readFile(fragPath);
    const char* vSrc = vertSrc.c_str();
    const char* fSrc = fragSrc.c_str();

    GLuint vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert, 1, &vSrc, nullptr);
    glCompileShader(vert);

    GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag, 1, &fSrc, nullptr);
    glCompileShader(frag);

    // Check errors
    GLint ok;
    char log[512];
    glGetShaderiv(vert, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(vert, 512, nullptr, log); fprintf(stderr, "VERT: %s\n", log); }
    glGetShaderiv(frag, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(frag, 512, nullptr, log); fprintf(stderr, "FRAG: %s\n", log); }

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);

    glDeleteShader(vert);
    glDeleteShader(frag);

    return prog;
}



void SkyRenderer::init() {
    // Fullscreen triangle (covers NDC [-1,1] with 3 verts — faster than quad)
    float verts[] = { -1,-1,  3,-1,  -1,3 };
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);
    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    m_Shader = loadShaderProgram("sky.vert", "sky.frag"); // your loader
}

void setUniform(GLuint shader, const char* name, const glm::mat4& mat) {
    glUniformMatrix4fv(glGetUniformLocation(shader, name), 1, GL_FALSE, glm::value_ptr(mat));
}

// 2. For your Sun Direction (vec3)
void setUniform(GLuint shader, const char* name, const glm::vec3& vec) {
    glUniform3fv(glGetUniformLocation(shader, name), 1, glm::value_ptr(vec));
}

// 3. For your Resolution (vec2)
void setUniform(GLuint shader, const char* name, const glm::vec2& vec) {
    glUniform2fv(glGetUniformLocation(shader, name), 1, glm::value_ptr(vec));
}

void SkyRenderer::render(const glm::mat4& invViewProj, glm::vec3 sunDir, glm::vec2 res) {
    glDepthFunc(GL_LEQUAL);  // sky at max depth
    glDepthMask(GL_FALSE);
    glUseProgram(m_Shader);

    setUniform(m_Shader, "u_InvViewProj", invViewProj);
    setUniform(m_Shader, "u_SunDir", sunDir);
    setUniform(m_Shader, "u_Resolution", res);

    glBindVertexArray(m_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
}