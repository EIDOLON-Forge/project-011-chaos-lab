# EIDOLON #011 — Chaos Lab

An interactive 3D visualization of the Lorenz attractor built with Python and Pygame.

Part of the **EIDOLON 500+ Project Journey**.

> Build. Experiment. Break. Repeat.

---

## What Is Chaos Lab?

Chaos Lab visualizes the famous **Lorenz attractor**, one of the best-known examples of a chaotic dynamical system.

It looks like a butterfly, but its shape comes from a small set of differential equations.

The system is deterministic.

There is no randomness required to generate the attractor.

Yet extremely small changes in the starting conditions can eventually produce dramatically different trajectories.

This is one of the key ideas behind chaotic systems.

---

# The Lorenz System

The simulation uses three differential equations:

```text
dx/dt = σ(y - x)

dy/dt = x(ρ - z) - y

dz/dt = xy - βz
```

The default parameters are:

```text
σ = 10
ρ = 28
β = 8/3
```

These are the classic Lorenz parameters that produce the familiar butterfly-shaped attractor.

---

# Numerical Integration

A computer cannot directly draw a continuous differential equation.

The system therefore needs to be integrated numerically.

Chaos Lab uses **fourth-order Runge-Kutta integration (RK4)**.

Conceptually:

```text
Current position
       ↓
Calculate several intermediate slopes
       ↓
Combine the slopes
       ↓
Calculate next position
       ↓
Repeat
```

This gives us a more accurate trajectory than a basic Euler step while remaining fast enough for real-time visualization.

---

# Features

- Real-time Lorenz attractor
- 3D perspective projection
- Multiple particles
- Long mathematical trails
- Camera rotation
- Camera zoom
- Automatic camera rotation
- Adjustable simulation speed
- Adjustable Lorenz parameters
- Parameter presets
- Multiple color palettes
- Glow
- Grid
- XYZ axes
- Pause / resume
- Random starting conditions
- Trail clearing
- Screenshot capture
- Procedural audio
- Telemetry HUD
- Smoke test

---

# Controls

| Input | Action |
| --- | --- |
| `SPACE` | Pause / resume |
| `R` | Reset |
| `J` | Randomize starting conditions |
| `P` | Change palette |
| `G` | Toggle glow |
| `T` | Toggle grid |
| `A` | Toggle axes |
| `V` | Toggle automatic camera rotation |
| `M` | Toggle audio |
| `S` | Save screenshot |
| `C` | Clear trails |
| `+` / `-` | Change simulation speed |
| `[` / `]` | Change trail length |
| `UP` / `DOWN` | Change ρ |
| `LEFT` / `RIGHT` | Change σ |
| `Z` / `X` | Change β |
| `1` | Classic parameters |
| `2` | Stable parameters |
| `3` | Chaotic parameters |
| Mouse drag | Rotate camera |
| Mouse wheel | Zoom |
| `ESC` | Quit |

---

# Parameter Presets

## Classic

```text
σ = 10
ρ = 28
β = 2.667
```

Produces the familiar Lorenz butterfly.

## Stable

```text
σ = 10
ρ = 20
β = 2.667
```

Produces significantly different dynamics.

## Chaotic

```text
σ = 14
ρ = 35
β = 3
```

Produces a more aggressive chaotic system.

---

# Why Does It Look Like a Butterfly?

The Lorenz system has regions in which trajectories are attracted toward different parts of the state space.

The trajectory repeatedly moves between these regions.

When projected into three dimensions, this produces the characteristic two-lobed structure.

The path never simply settles into a fixed point.

Instead, it remains bounded while continuing to evolve.

---

# Chaos Is Not Randomness

One of the interesting properties of the Lorenz system is that it is deterministic.

Given:

```text
same parameters
+
same initial conditions
```

the trajectory is completely determined.

But changing the initial condition by a tiny amount can eventually produce a significantly different trajectory.

This is called:

**sensitive dependence on initial conditions.**

---

# 3D Rendering

The Lorenz equations generate coordinates:

```text
x
y
z
```

The program then transforms those coordinates according to the virtual camera:

```text
Lorenz coordinates
        ↓
Yaw rotation
        ↓
Pitch rotation
        ↓
Perspective projection
        ↓
Screen coordinates
```

This creates the 3D effect without requiring a full 3D engine.

---

# Why RK4?

A simple Euler integrator could calculate:

```text
x_next = x + dx * dt
```

and similarly for `y` and `z`.

However, chaotic systems are sensitive to numerical errors.

Chaos Lab therefore uses fourth-order Runge-Kutta integration.

RK4 evaluates the system multiple times during each step and combines those evaluations into a more accurate estimate.

---

# Technology

## Python

Application logic and mathematical simulation.

## Pygame

Used for:

- Window management
- Rendering
- Input
- Timing
- Audio

No external graphics engine is required.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/EIDOLON-Forge/project-011-chaos-lab.git
```

Enter the directory:

```bash
cd project-011-chaos-lab
```

Create a virtual environment:

```bash
python -m venv .venv
```

Git Bash:

```bash
source .venv/Scripts/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

# Run

```bash
python main.py
```

---

# Smoke Test

Run:

```bash
python main.py --smoke-test
```

The program performs numerical simulation steps and verifies that the resulting coordinates remain finite.

Expected output:

```text
Chaos Lab smoke test: OK
Lorenz parameters: sigma=10.0, rho=28.0, beta=2.666667
Trail points: 101
```

---

# Project Architecture

```text
project-011-chaos-lab/
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

The project intentionally remains self-contained.

---

# Learning Goals

This project introduces:

### Mathematics

- Differential equations
- Dynamical systems
- Chaos theory
- Numerical integration
- State-space visualization

### Programming

- Simulation loops
- Numerical methods
- Data structures
- Real-time updates
- Performance management

### Graphics

- 3D coordinates
- Rotation matrices
- Perspective projection
- Camera systems
- Trail rendering

---

# EIDOLON Progression

Chaos Lab continues the mathematical and visual progression of EIDOLON.

```text
Simple physics
      ↓
Particles
      ↓
3D projection
      ↓
Cellular automata
      ↓
Fractals
      ↓
GPU graphics
      ↓
Chaotic dynamical systems
```

The important part is not simply making the visuals more complicated.

Each project introduces another useful concept.

---

# Future Extensions

The system can later be expanded into:

- Lorenz variations
- Rossler attractor
- Thomas attractor
- Double pendulum
- N-body simulations
- Strange attractors
- Particle fields
- Fluid simulations
- GPU-based differential equations
- 3D ray-marched mathematical systems

These concepts are intentionally reserved for future EIDOLON projects.

---

# Project Status

**Project:** EIDOLON #011

**Name:** Chaos Lab

**Subject:** Lorenz Attractor

**Category:** Mathematics / Simulation / Graphics

**Language:** Python

**Framework:** Pygame

**Status:** Active EIDOLON project

---

# EIDOLON

EIDOLON is a long-term experiment in seeing how far one person can take coding by building something new again and again.

500+ projects.

One project at a time.

Build.

Experiment.

Break.

Repeat.
