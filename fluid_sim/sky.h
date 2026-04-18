#pragma once
#include<glm/glm.hpp>
#include<glad/glad.h>

class SkyRenderer {
public:
    void init();
    void render(const glm::mat4& invViewProj, glm::vec3 sunDir, glm::vec2 res);
   // void destroy();

    glm::vec3 sunDir = glm::normalize(glm::vec3(0.4f, 0.6f, 0.3f));

private:
    GLuint m_VAO, m_VBO, m_Shader;
};


extern SkyRenderer sky;