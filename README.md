# 🌊 SPH Fluid Simulation

A **real time Smoothed Particle Hydrodynamics (SPH) fluid ** simulation built for realistic fluid behavior using **GPU acceleration (CUDA)** and spatial hashing for fast neighbor searches.

This project focuses on **real‑time fluid physics**, tunable physical parameters, and scalability for thousands of particles — designed for experimentation, learning, and performance.


## notice
not perfect yet!
so expect errors or bugs and feel free to give feedback
-----------------------
<img width="733" height="289" alt="Screenshot 2026-02-26 204642" src="https://github.com/user-attachments/assets/c0245bb1-d76b-4450-b117-6a3be69070de" />
<img width="751" height="337" alt="Screenshot 2026-02-26 204618" src="https://github.com/user-attachments/assets/b0481854-770f-45e4-bed3-7042e9e9934b" />



https://github.com/user-attachments/assets/f57c37d3-220c-47be-bcfe-0d024ce1de9c

-------------------------
---
## comtrols
wasd for camera movement
shift and space to change height movement

## 🚀 Features

* ⚡ **GPU‑accelerated SPH (CUDA)**
* 🧠 **Spatial grid / hashing** for fast neighbor lookup

* 💧 Realistic **pressure, density, viscosity & surface tension**
* 📦 Configurable **bounding box with friction & damping**
* 🧪 Fully **tunable simulation parameters**
* 📊 Debug tools for density, pressure & neighbor counts
* 🎮 Designed for **real‑time interactive simulation**

---

## 🧬 Core Simulation Model

This implementation follows a **force‑based SPH pipeline**:

### 1️⃣ Density Computation

Each particle samples nearby neighbors using a smoothing kernel:

* Rest Density
* Kernel Radius `h`
* Mass per particle

### 2️⃣ Pressure Calculation

Pressure is derived using an equation of state:

* Gas constant `K`
* Gamma / stiffness
* standard pressure equation
* Density error from rest density

### 3️⃣ Force Evaluation

Particles receive forces from:

* Pressure gradients
* Viscosity forces
* Surface tension
* Gravity
* Boundary collisions

### 4️⃣ Integration

Particle motion is updated via:

* Velocity update
* Position update
* Damping & restitution

---

## 🧪 Adjustable Parameters

| Parameter        | Purpose                 |
| ---------------- | ----------------------- |
| `h`              | Smoothing radius        |
| `cellSize`       | Grid resolution         |
| `K`              | Pressure stiffness      |
| `restDensity`    | Target fluid density    |
| `alphaVisc`      | Linear viscosity        |
| `betaVisc`       | Quadratic viscosity     |
| `surfaceTension` | Surface smoothing       |
| `gravity`        | External force          |
| `restitution`    | Bounce strength         |
| `friction`       | Wall sliding resistance |
| `damping`        | Energy loss             |

---

## 🧠 Key Goals

* Stable **fluid compression without explosive pressure**
* Balanced **neighbor count for natural water behavior**
* High‑speed GPU performance for **10k–100k+ particles**
* Parameter‑driven realism instead of hacks

---

## 🏗️ Architecture Overview

```
Particles
 ├─ Density Kernel Pass
 ├─ Pressure Solve Pass
 ├─ Force Accumulation Pass
 ├─ Integration Pass
 └─ Collision Handling

Grid Hash
 ├─ Cell indexing
 └─ Neighbor search
```

---

## ⚙️ Build & Run
 
### CUDA Build Example
cuda is used as a lib which is compiled befor with the help of build_cuda.bat file
 download the whole repo and in the build cuda file edit the compute.cu address and compile before the whole project




> Recommended GPU: nivida gpu 
* if faced error in cuda compilation or runtime error then tweak the arch sm_ in the build_cuda.bat with your gpu arch like sm_75,sm_85 etc based on gpu series
---

## 🐛 Debug & Diagnostics

* Print neighbor counts
* Inspect density & pressure


---

## 📈 Performance Tips

* Tune `h` and `cellSize` together
* Keep average neighbors between **20–60**
* Avoid extreme `K` values (causes pressure spikes)
* Clamp max velocity to prevent tunneling

---

## 🎯 Planned Improvements

* 🌪️ Vorticity confinemen

* 🧵 Async GPU compute pipeline
* 🎥 Real‑time visualization UI

---

## 📜 License

Open‑source for learning & experimentation. Modify freely.

---

## ✨ Credits

Built with passion for **fluid physics, GPU compute, and simulation engineering**.

feel free to contibute 

