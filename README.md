# Dynamical Systems — Numerical Experiments and Learning Notes

This repository is a growing collection of Python simulations, interactive tools, and Jupyter notebooks developed during my independent study of **dynamical systems**. Its purpose is to connect the mathematical theory of differential equations with numerical solutions, phase-space visualisation, and computational experiments.

The projects begin with elementary autonomous systems and extend to nonlinear oscillators, population dynamics, neuronal excitability, and a model of an oscillating chemical reaction. The emphasis is on understanding *why* a system behaves as it does: locating equilibria, examining local stability, following trajectories from different initial conditions, and comparing linear approximations with nonlinear dynamics.

## Topics explored

- **Differential equations and numerical integration:** initial-value problems, coupled ODEs, time evolution, and adaptive integration using SciPy.
- **Equilibria and stability:** fixed points, Jacobian matrices, eigenvalues, eigenvectors, and local linearisation.
- **Geometric analysis:** phase lines, phase portraits, direction fields, nullclines, and trajectories in two- and three-dimensional state spaces.
- **Nonlinear behaviour:** nonlinear damping, self-sustained oscillations and limit cycles, predator–prey dynamics, and excitable systems.
- **Computational investigation:** varying parameters and initial conditions, comparing analytical and numerical solutions, and building interactive GUIs.

## Projects

| Folder / file | Model or exercise | Main concepts and methods |
| --- | --- | --- |
| [`Prey_Predator/`](Prey_Predator/) | Lotka–Volterra predator–prey model | Coupled nonlinear ODEs, nullclines, equilibria, Jacobian eigenanalysis, phase portraits and interactive simulation. |
| [`Damped_Oscillator/`](Damped_Oscillator/) | Damped harmonic oscillator | Linear systems, damping, eigenvalues, phase portraits, and comparison of numerical solutions with the matrix-exponential solution. |
| [`Van_Der_Pol_Oscillator/`](Van_Der_Pol_Oscillator/) | Van der Pol oscillator | Nonlinear damping, oscillations, phase-space trajectories, and nonlinear versus locally linearised dynamics. |
| [`Fitzhugh_Nagumo_neuron/`](Fitzhugh_Nagumo_neuron/) | FitzHugh–Nagumo neuronal model | Excitability, nullclines, equilibrium calculations, local stability, and nonlinear/linearised comparisons. |
| [`Belousov–Zhabotinsky_Oregonator/`](Belousov–Zhabotinsky_Oregonator/) | Simplified Oregonator model of the Belousov–Zhabotinsky reaction | Three coupled nonlinear ODEs, time-series analysis, and 2D/3D phase-space trajectories. |
| `Intro_Dyanmical_Systems.ipynb` | Introductory notebook | One-dimensional autonomous systems, fixed points, RC-circuit dynamics, linear systems, eigenvectors and analytical/numerical comparisons. |

Each model folder contains its own `README.md` with its equations, parameter definitions, file descriptions, and usage instructions. The repository is intended to expand as I study more systems and numerical techniques.

## Software and dependencies

The projects are written in **Python** and use the following packages:

| Package | Role in the projects |
| --- | --- |
| **NumPy** | Arrays, vectorised calculations, grids, polynomial roots, and basic linear algebra. |
| **SciPy** | Solving ODE initial-value problems (`solve_ivp`), matrix exponentials, and eigenvalue calculations. |
| **Matplotlib** | Time-series plots, phase portraits, vector fields, and 3D trajectories. |
| **JAX** | Automatic differentiation, especially `jax.jacfwd` for computing Jacobian matrices in stability and linearisation exercises. |
| **PyQt5** | Desktop interfaces for changing model parameters, initial conditions, and simulation duration. |
| **pandas**, **tabulate** | Formatted numerical summaries in the Lotka–Volterra command-line script. |
| **Jupyter** | Running the introductory notebook. |

**JAX is used for differentiation, not because these small simulations require GPU acceleration.** In particular, the damped-oscillator, Van der Pol, and FitzHugh–Nagumo scripts use `jax.jacfwd` to compute Jacobians. A CPU-only JAX installation is sufficient to run them. The Lotka–Volterra and Oregonator scripts do not require JAX.

### Recommended installation: isolated environment

To avoid dependency conflicts with unrelated scientific Python projects, create a **separate environment** rather than installing JAX into an existing environment.

For example, with Conda:

```bash
conda create -n dynamical-systems python=3.12 -y
conda activate dynamical-systems
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pandas tabulate PyQt5 notebook ipykernel
```

Choose **one** of the following JAX installations.

**CPU (recommended for these projects):**

```bash
python -m pip install --upgrade jax
```

**NVIDIA GPU on a compatible Linux system (optional, CUDA 12):**

```bash
python -m pip install --upgrade "jax[cuda12]"
```

The CUDA 12 installation requires compatible NVIDIA hardware and drivers. It is optional: the models do not need a GPU. See the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for current hardware, driver, and CUDA requirements. CUDA 13 may be preferable for newer supported hardware.

Verify the installation:

```bash
python -m pip check
python -c "import jax; print('JAX devices:', jax.devices())"
```

Seeing `CpuDevice(...)` is normal for a CPU installation; a correctly configured GPU installation should report a CUDA device.

### Running the projects

Run scripts from their respective project folders. For example:

```bash
cd prey_predator
python LV_Model.py
python LV_GUI.py
```

Other common entry points are:

| Project | Standalone script | Interactive GUI |
| --- | --- | --- |
| Damped oscillator | `Damped_Oscillator.py` | `Damped_Oscillator_GUI.py` |
| Van der Pol | `VDP.py` | `VDP_GUI.py` |
| FitzHugh–Nagumo | `FitzHugh_Nagumo.py` | `FHN _GUI.py` |
| Oregonator | `bz.py` | — |

The FitzHugh–Nagumo GUI filename contains a space before the underscore in the original uploaded code; quote it when running it (`python "FHN _GUI.py"`), or rename the script and update its project README.

To open the introductory notebook, start Jupyter from the repository directory:

```bash
jupyter notebook
```

The PyQt5 applications require a desktop environment with Qt support. Most standalone scripts use editable parameter values and initial conditions near the top of their source files; the GUI versions provide input fields or sliders. Consult the README in each subfolder for exact defaults and formats.

## Notes on scope

These are **educational and exploratory implementations**, not validated research software or general-purpose simulation packages. Most projects focus on a small number of illustrative parameter choices. Jacobian eigenanalysis concerns **local** behaviour near equilibria; it does not by itself establish global stability. Numerical trajectories also depend on solver settings and the chosen model parameters.

As the repository grows, I plan to add further model problems and numerical experiments to connect dynamical-systems theory with computational scientific modelling.
