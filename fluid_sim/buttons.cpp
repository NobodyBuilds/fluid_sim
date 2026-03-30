#include "buttons.h"
#include "settings.h"
#include "GLFW/glfw3.h"
#include "main.h"
#include "imgui_impl_glfw.h"

const unsigned int screenWidth = 1800;
const unsigned int screenHeight = 900;

static bool predown = false;
static bool f11down = false;
static bool xdown = false;
static bool runstatedown = false;



extern "C" void buttons(GLFWwindow* window){
	//sim pause logic
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
        restartSimulation();
   
    bool down = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;

    if (down && !predown)
    {
        if (settings.nopause == true) {
            settings.nopause = false;
        }
        else if (settings.nopause == false) {
            settings.nopause = true;
        }
    }
    predown = down;

    //fullcreren button
    bool f11 = glfwGetKey(window, GLFW_KEY_F11) == GLFW_PRESS;
    if (f11 && !f11down) {
        if (glfwGetWindowMonitor(window)) {
            glfwSetWindowMonitor(window, nullptr, 100, 100, screenWidth, screenHeight, 0);
        }
        else {
            GLFWmonitor* primary = glfwGetPrimaryMonitor();
            const GLFWvidmode* mode = glfwGetVideoMode(primary);
            glfwSetWindowMonitor(window, primary, 0, 0, mode->width, mode->height, mode->refreshRate);
        }
    }
    f11down = f11;
   //debug menu
    bool x = glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS;
    if (x && !xdown)
    {
        if (settings.debug == true) {
            settings.debug = false;
        }
        else if (settings.debug == false) {
            settings.debug = true;
        }
    }
    xdown = x;

    


}