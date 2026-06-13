# CUDA SPH Fluid Simulation

A real-time Smoothed Particle Hydrodynamics (SPH) fluid simulation running entirely on the GPU. Built from mathematical first principles using CUDA and OpenGL interop — no game engines, no pre-built solvers.

Designed for curiosity, and learning GPU-accelerated physics. Scales comfortably from 10k to 200k+ particles at interactive frame rates real-time also capable of running large simulations upto 1 million particles for non real-time renders to use it turn on "simulated" option from ui .

---

## Requirements

- Windows (only supported platform)
- NVIDIA GPU (CUDA-capable)


---

## Build Instructions

There is no CMake. The build is two steps:

**Step 1 — Compile the compute engine**

Open `build_cuda.bat` and verify the path to `compute.cu` is correct for your machine. Run the `.bat` file. This produces `compute.lib`, which contains all CUDA kernels.

**Step 2 — Build the main project**

Compile the rest of the project normally (MSVC). Link against `compute.lib` and the standard dependencies (CUDA runtime, GLFW, GLAD, ImGui).

If CUDA compilation fails, the most common fix is adjusting the `sm_` architecture flag in `build_cuda.bat` to match your GPU.

---

## Controls

| Input              | Action                  |
| ------------------ | ----------------------- |
| LMB drag           | Rotate camera           |
| Scroll wheel       | Zoom in / out           |
| W / A / S / D      | Move camera             |
| Q                  | Move camera up          |
| E                  | Move camera down        |
| Space              | Pause / unpause         |
| K                  | Restart simulation      |
| H                  | Toggle UI visibility    |
| X                  | Toggle debug overlay    |

---

## Features

- proper ui with multiple tuneable parameters in multiple tabs
- 3 rendering styles(in rendering tab)
   1=native particles with velocity rgb(also allows selecting color for particle)
   2=screen space effect for a blured fluid (looks decent)
   3= ray marching fluid with great visuals(fps may fluctuate)
- "simulated" option in ui allows switching from real-time to frame to frame.
- tuneable fluid boundaries from ui in world tab(box can be hiiden from ui).
- "wave generation" option in ui help to form waves for better visuals.
- floor size,color can be tuned live fron ui
- dymanic sky with moveable sun
- Debug overlay showing per-particle density, pressure, and neighbor counts
-dynamically emit particles with reallocation live (using preallocated buffer for emitter).
---

## Core Simulation Model


- The simulation runs a standard force-based WCSPH pipeline each frame, fully on the GPU
- uses trait eos for pressure
- xsph for velocity smoothing
- viscosity for motion damping
- velocity verlet intigration

---


## Debug Overlay

Press `X` to enable the debug overlay. Displays avg per-particle:

- Neighbor count
- Estimated density
  

Useful for diagnosing clumping, pressure spikes.

---

## License

Open-source. Free to use, modify, and learn from.

---

## Credits

Built from scratch in CUDA C++ and OpenGL — no game engine, no pre-built physics library.  
