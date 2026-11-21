
# FitzHugh–Nagumo Neuron Model

This project explores the FitzHugh–Nagumo model, a two-variable nonlinear system used to study neuronal excitability. It includes numerical trajectories, nullcline and equilibrium analysis, comparison with a local linearised model, and an interactive PyQt5 simulator. The work forms part of my independent study of dynamical systems.

## Mathematical Model

The model describes a fast excitation variable $v$ and a slower recovery variable $w$:

```math
\frac{dv}{dt} = v - \frac{v^3}{3} - w + I_{\mathrm{ext}}
```

```math
\frac{dw}{dt} = \varepsilon(v+a-bw)
```

Here:

- $v$ represents the excitation or membrane-potential variable.
- $w$ represents the slower recovery variable.
- $I_{\mathrm{ext}}$ represents the applied external input.
- $a$ and $b$ control the recovery dynamics.
- $\varepsilon$ determines the relative recovery timescale.

**In the Python scripts, $\varepsilon$ is represented by the variable `e`.**

### Nullclines and Equilibrium Points

The two nullclines are obtained by setting the corresponding time derivatives to zero:

```math
w = v - \frac{v^3}{3} + I_{\mathrm{ext}}
```

```math
w = \frac{v+a}{b}, \qquad b \ne 0
```

Their intersections represent equilibrium points.

The Jacobian matrix of the system is:

```math
J(v,w) =
\begin{pmatrix}
1-v^2 & -1 \\
\varepsilon & -\varepsilon b
\end{pmatrix}
```

The eigenvalues of the Jacobian evaluated at an equilibrium point provide information about its local stability. Linearisation gives a local approximation to the nonlinear dynamics and may differ from the original system away from the equilibrium.

## Numerical Methods and Analysis

The project includes:

1. **Numerical Integration**
   - Solving the coupled nonlinear ODEs using SciPy's `solve_ivp`.
   - Using the default adaptive RK45 integration method.
   - Comparing trajectories from two initial conditions.

2. **Phase-Plane Analysis**
   - Plotting nonlinear phase-plane trajectories.
   - Visualising normalised vector fields.
   - Calculating and plotting both nullclines.
   - Identifying an equilibrium point.

3. **Equilibrium and Stability Analysis**
   - Finding a real equilibrium using `numpy.roots`.
   - Computing the Jacobian using JAX automatic differentiation (`jacfwd`).
   - Calculating Jacobian eigenvalues and eigenvectors in the GUI.
   - Comparing nonlinear trajectories with a Jacobian-based linear system.

4. **Scientific Visualisation**
   - Plotting time evolution of the excitation and recovery variables.
   - Visualising the derivative surfaces `dv/dt` and `dw/dt` in three dimensions.
   - Comparing nonlinear and linearised phase portraits.

5. **Interactive Simulation**
   - Developing a GUI using PyQt5.
   - Allowing changes to model parameters and initial conditions.
   - Adjusting simulation duration.
   - Displaying numerical stability information and updated plots.

## Implementation

### Python Libraries

- **NumPy:** Array operations, polynomial roots, and numerical calculations.
- **SciPy:** Numerical integration and eigenvalue calculations.
- **Matplotlib:** Time-series, phase-plane, vector-field, and 3D visualisation.
- **JAX:** Automatic differentiation for Jacobian computation.
- **PyQt5:** Interactive graphical user interface.

## Project Files and Inputs

### `FHN.py` — Standalone Simulation

This script performs numerical integration, equilibrium analysis, phase-plane visualisation, and comparison with a linearised model.

The parameters and initial conditions can be modified directly near the beginning of the file:

```python
Iext = 0.5
a = 0.7
b = 0.8
e = 0.08

t_span = (0.0, 200.0)
initial_conditions = [[-1.5, -1.0], [0.5, 0.0]]
```

**Parameter descriptions:**

- `Iext`: Applied external input.
- `a`: Recovery offset parameter.
- `b`: Recovery coupling parameter.
- `e`: Relative recovery timescale factor.
- `t_span`: Beginning and end of the simulation interval.
- `initial_conditions`: List of initial excitation and recovery values.

Each initial condition follows the format:

```python
[v_initial, w_initial]
```

For example:

```python
initial_conditions = [[-1.5, -1.0], [0.5, 0.0]]
```

This runs the nonlinear system for two different initial states.

The script produces six plots: two derivative surfaces, nonlinear time series, nonlinear phase portrait, linearised time series, and a linearised phase portrait.

### `FHN_GUI.py` — Interactive Simulator

The PyQt5 interface allows users to change parameters and initial conditions without editing the source code.

| Input | Meaning | Default |
| --- | --- | --- |
| `a` | Recovery offset | `0.7` |
| `b` | Recovery coupling | `0.8` |
| `e` | Relative recovery timescale | `0.08` |
| `Iext` | Applied external input | `0.5` |
| Initial condition 1 | First `v, w` pair | `-1.5, -1.0` |
| Initial condition 2 | Second `v, w` pair | `0.5, 0.0` |
| Simulation end time | Slider ranging from 50 to 500 | `200` |

**Entering initial conditions:**

Each initial condition should be entered as two comma-separated numbers, without square brackets.

Example:

```text
-1.5, -1.0
```

The GUI provides:

- Two editable sets of initial conditions.
- Four adjustable model parameters.
- A simulation-duration slider.
- Six graphical outputs.
- Equilibrium coordinates, Jacobian matrix, eigenvalues, and eigenvectors.

Click **Run Simulation** to execute the model with the selected parameters.

## Installation

Install the required Python libraries:

```bash
pip install numpy scipy matplotlib jax PyQt5
```

## Usage

### Run the Standalone Simulation

```bash
python FHN.py
```

### Launch the Interactive GUI

```bash
python FHN_GUI.py
```

The GUI requires a desktop environment with Qt support.

## Learning Outcomes

This project provided practical experience in:

- Solving coupled nonlinear ordinary differential equations.
- Constructing phase portraits and vector fields.
- Calculating and interpreting nullclines and equilibrium points.
- Applying Jacobian-based local stability analysis.
- Comparing nonlinear dynamics with local linear approximations.
- Using automatic differentiation for mathematical analysis.
- Developing interactive scientific simulations using Python.

## Scope and Limitations

The implementation identifies and displays one real equilibrium obtained from the cubic equation. For parameter settings where multiple real equilibria exist, the scripts do not automatically investigate all of them.

The Jacobian-based linear approximation is intended to describe behaviour near equilibrium and should not be interpreted as a global approximation of the nonlinear system.
