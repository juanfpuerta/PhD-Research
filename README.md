# Forecasting Catastrophic Scenarios in LEO via Symplectic N-Body Integrators
### AI-Enhanced Hybrid Framework for Massive Space Debris Population Dynamics

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Project%20Site-blue?logo=github)](https://juanfpuerta.github.io/phd-research/)
[![Overleaf Sync](https://img.shields.io/badge/Overleaf-Git%20Bridge-green?logo=overleaf)](https://www.overleaf.com)
[![License: MIT / CC BY 4.0](https://img.shields.io/badge/License-MIT%20%2F%20CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/CI%2FCD-LaTeX%20Auto--Build-brightgreen?logo=githubactions)](.github/workflows/deploy-pages.yml)
[![Institution](https://img.shields.io/badge/Institution-Universidad%20de%20Antioquia-006241?logo=google-earth)](https://www.udea.edu.co)

---

## 📌 Overview

This repository hosts the computational codebase, thesis manuscript sources, and dynamic documentation for the doctoral research project conducted by **Juan Francisco Puerta-Ibarra** within the **GIMEL Research Group** at **Universidad de Antioquia (UdeA)**.

Long-term forecasting of orbital fragmentation and Kessler syndrome cascade risks in Low Earth Orbit (LEO) is fundamentally bottlenecked by standard numerical integration architectures:
1. **The Dissipative Barrier:** Symplectic schemes natively preserve symplectic 2-forms ($d\mathbf{p} \wedge d\mathbf{q}$) in conservative Hamiltonian systems, but break down when introducing non-conservative atmospheric drag and Solar Radiation Pressure (SRP).
2. **Secular Error Accumulation:** Classical explicit schemes (Cowell, Runge-Kutta RK8/9) accurately capture dissipative forces but suffer from severe secular energy drift and prohibitive computational costs ($\mathcal{O}(N^2)$) over decadal propagation arcs.

This project delivers an **AI-enhanced, high-fidelity hybrid semi-symplectic computational framework** that bridges this divide—preserving near-Hamiltonian phase-space invariants while accelerating multi-decade forecasting across massive autonomous debris clouds ($N \ge 50,000$).

---

## 🧬 Framework Architecture & Mathematical Formulation

The engine couples an optimized C-based symplectic backbone (adapted from `WHFast` / `REBOUND`) with specialized deep learning surrogates via a **second-order symmetric Strang splitting sequence**:

```
                       [ Initial State Vector Cloud (N >= 50,000) ]
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │      Phase 1: Dissipative Half-Kick Φ_P      │
                    │   a_diss(drag, SRP) kick: Δt/2 update        │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │       Phase 2: Symplectic Drift Flow Φ_G     │
                    │   Conservative J2/J4 Geopotential Field      │
                    │   WHFast Symplectic Map: Δt full step        │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      Phase 3: Dissipative Half-Kick Φ_P      │
                    │   Recalculated a_diss kick: Δt/2 update      │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                             [ Time Check: t + Δt < T_max ]
                                    │               ▲
                             No     │               │  Yes
                             ───────┴───────────────┘
```

The composite flow operator $\Phi(\Delta t)$ guarantees second-order temporal accuracy $\mathcal{O}(\Delta t^2)$:

$$\Phi(\Delta t) \approx \Phi_P\left(\frac{\Delta t}{2}\right) \circ \Phi_G(\Delta t) \circ \Phi_P\left(\frac{\Delta t}{2}\right)$$

### Key Components

* **Semi-Symplectic Core ($\Phi_G$ & $\Phi_P$):** Integrates conservative geopotential harmonics ($J_2, J_4$) symplectically, while interleaving non-conservative atmospheric drag and SRP without artificial secular energy decay.
* **PINN Dynamic Thermospheric Surrogates:** Trains Physics-Informed Neural Networks (PINNs) on NRLMSISE-00 / Jacchia-Roberts data, dynamically resolving solar flux ($F_{10.7}$) and geomagnetic storms ($K_p/A_p$) with $>10\times$ evaluation speedup and $<2\%$ density error.
* **Spatio-Temporal Transformer / LSTM Screening:** Bypasses combinatorial all-pairs conjunction testing by dynamically detecting high-probability collision clusters within the 50,000-particle cloud in real time.
* **Backward Reinforcement Learning (RL) Inversion:** Exploits time-reversibility characteristics to trace decaying debris clouds backward in time, recovering parent-satellite breakup epochs and fragmentation coordinates.

---

## 📊 Quantitative Benchmarks (AAS 25-787 Baseline & Targets)

| Metric / Parameter | Industry Standard (Cowell / RK8) | Preliminary Baseline (AAS 25-787) | Doctoral Target Framework |
| :--- | :--- | :--- | :--- |
| **Simulated Particles ($N$)** | $\sim 10^3$ | $20,000$ | $\ge 50,000$ |
| **Relative Energy Drift ($|\Delta E / E_0|$)** | Diverges secularly | $10^{-8} \le \|\Delta E / E_0\| \le 10^{-7}$ | Strictly bounded $\le 10^{-7}$ |
| **Secular Nodal Precession Error** | Numerical damping | $< 0.005\%$ vs analytical $J_2$ | $< 0.005\%$ long-arc stability |
| **Benchmark Agreement (NASA GMAT)** | Baseline reference | Sub-kilometer deviation (24 h) | Sub-kilometer cross-tool verification |
| **Scalability Complexity** | Prohibitive $\mathcal{O}(N^2)$ | Asymptotic power-law $\mathcal{O}(N^{1.60})$ | Linear / sub-quadratic ($\le \mathcal{O}(N \log N)$) |

---

## 🗂️ Repository Structure

```text
phd-research/
├── .github/
│   └── workflows/
│       ├── compile-latex.yml     # Automated build of thesis PDF from Overleaf sync
│       └── deploy-pages.yml      # Builds and deploys documentation to GitHub Pages
├── docs/                         # GitHub Pages static site sources (Jekyll / HTML)
│   ├── index.md                  # Landing page with interactive thesis summary
│   ├── methodology.md            # Mathematical formulation & Strang splitting
│   ├── benchmarks.md             # GMAT cross-validation & energy plots
│   └── assets/                   # Figures, diagrams, and simulation media
├── src/                          # Core simulation codebase
│   ├── c_core/                   # High-performance Strang-splitting engine (C/CUDA)
│   ├── pinn/                     # Thermospheric surrogate network models (PyTorch)
│   ├── conjunction/              # Transformer / LSTM screening modules
│   └── inversion/                # Backward RL trajectory reconstruction
├── manuscript/                   # Git-bridged Overleaf thesis workspace
│   ├── main.tex                  # Primary LaTeX root document
│   ├── chapters/                 # Modular chapters (01_intro.tex, 02_sota.tex, ...)
│   ├── figures/                  # Publication-grade vector graphics
│   └── references.bib            # Master BibTeX reference library
├── data/                         # Initial conditions, TLE subsets, and validation logs
└── README.md                     # Repository landing page
```

---

## 🔄 Overleaf & GitHub Pages Synchronization

This repository uses a two-way synchronization between **Overleaf**, **GitHub**, and **GitHub Pages**:

```
 ┌─────────────────┐       git push / pull        ┌─────────────────┐
 │                 │ <──────────────────────────> │                 │
 │  Overleaf Repo  │                              │  GitHub Master  │
 │ (Writing/LaTeX) │                              │ (Code & Paper)  │
 └─────────────────┘                              └────────┬────────┘
                                                           │
                                             GitHub Action │ CI / CD Trigger
                                                           ▼
                                                  ┌─────────────────┐
                                                  │  GitHub Pages   │
                                                  │ (Public Report) │
                                                  └─────────────────┘
```

### 1. Linking Overleaf to GitHub
1. In your Overleaf thesis project, navigate to **Menu** $\rightarrow$ **Sync** $\rightarrow$ **GitHub**.
2. Select this repository (`juanfpuerta/phd-research`) and bind the target branch (`main`).
3. Commit pushes from Overleaf will trigger automated GitHub Actions to rebuild documents and refresh documentation assets.

### 2. Automated PDF Compilation Workflow (`.github/workflows/compile-latex.yml`)
When changes are pushed to `manuscript/`, GitHub Actions compiles `main.tex`, attaches the compiled `thesis_proposal.pdf` to GitHub Releases, and copies the generated PDF into `docs/assets/` for live download on the GitHub Pages site.

---

## 📅 Phased Research Chronogram (24-Month Plan)

| Milestone / Work Package | M1–M4 | M5–M8 | M9–M12 | M13–M16 | M17–M20 | M21–M24 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WP1: Strang Splitting Integration Core** | 🟢 | | | | | | In Progress |
| **WP2: HPC Cloud Scaling ($\ge 50\text{k}$ particles)** | | ⚪ | | | | | Scheduled |
| **WP3: PINN Thermospheric Surrogate Training** | | | ⚪ | | | | Scheduled |
| **WP4: Conjunction Transformer/LSTM Logic** | | | | ⚪ | | | Scheduled |
| **WP5: Backward RL Path Reconstruction** | | | | | ⚪ | | Scheduled |
| **WP6: Cross-GMAT Validation & Defense** | | | | | | ⚪ | Scheduled |

---

## 💻 Quick Start & Local Execution

### Prerequisites
* GCC/Clang with OpenMP support (or CUDA Toolkit $\ge 12.0$ for GPU acceleration)
* Python $\ge 3.10$ with `torch`, `numpy`, `scipy`, `matplotlib`
* TeX Live $\ge 2024$ (for compiling local LaTeX sources)

### Build & Run the Semi-Symplectic Engine
```bash
# Clone the repository
git clone https://github.com/juanfpuerta/phd-research.git
cd phd-research

# Compile the C integration core
make -C src/c_core all

# Run test propagation with synthetic fragment cloud (N=20,000)
./src/c_core/bin/strang_propagator --particles 20000 --step 60 --days 30 --output data/output/
```

---

## 📜 Publications & Citation

If you utilize this framework, reference datasets, or preliminary benchmark configurations, please cite the foundational conference paper:

```bibtex
@inproceedings{puerta2025aas,
  author    = {Puerta-Ibarra, Juan Francisco},
  title     = {High-Fidelity Symplectic Propagation for Massive Space Debris Populations in Low Earth Orbit},
  booktitle = {AAS/AIAA Astrodynamics Specialist Conference},
  series    = {Advances in the Astronautical Sciences},
  volume    = {AAS 25-787},
  year      = {2025}
}
```

---

## 🏛️ Research Group & Contact

**Author:** Juan Francisco Puerta-Ibarra  
**Affiliation:** GIMEL Research Group, Faculty of Engineering, Universidad de Antioquia (UdeA), Medellín, Colombia  
**Email:** `juanf.puerta@udea.edu.co`  
**Institutional Portal:** [Universidad de Antioquia — Faculty of Engineering](https://www.udea.edu.co)
