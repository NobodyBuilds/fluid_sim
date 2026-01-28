# 🌊 SPH Fluid Simulation

A **high‑performance Smoothed Particle Hydrodynamics (SPH)** simulation built for realistic fluid behavior using **GPU acceleration (CUDA)** and spatial hashing for fast neighbor searches.

This project focuses on **real‑time fluid physics**, tunable physical parameters, and scalability for thousands of particles — designed for experimentation, learning, and performance.

---

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

```bash
nvcc -O3 source/compute.cu -o build
```



> Recommended GPU: RTX 2000+ series or higher
* reste the gpu sm_ for campatibility
---

## 🐛 Debug & Diagnostics

* Print neighbor counts
* Inspect density & pressure
* Detect NaNs or unstable kernels
* Compare CPU vs GPU behavior

---

## 📈 Performance Tips

* Tune `h` and `cellSize` together
* Keep average neighbors between **20–60**
* Avoid extreme `K` values (causes pressure spikes)
* Clamp max velocity to prevent tunneling

---

## 🎯 Planned Improvements

* 🧩 Position‑Based SPH mode
* 🌪️ Vorticity confinement
* 🫧 Foam & splash effects
* 🌊 Multi‑phase fluids
* 🧵 Async GPU compute pipeline
* 🎥 Real‑time visualization UI

---

## 📜 License

Open‑source for learning & experimentation. Modify freely.

---

## ✨ Credits

Built with passion for **fluid physics, GPU compute, and simulation engineering**.

feel free to contibute 
* issues

- low neibhor count
