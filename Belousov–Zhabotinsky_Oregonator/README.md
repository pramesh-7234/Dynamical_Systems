# Belousov–Zhabotinsky Reaction: Simplified Oregonator Model

This project investigates a simplified three-variable Oregonator model associated with the Belousov–Zhabotinsky (BZ) oscillating chemical reaction. Developed as part of my independent study of dynamical systems, it uses numerical integration to examine how three interacting state variables evolve over time and move through phase space.

## Mathematical model

The model is defined by three coupled nonlinear ordinary differential equations:

```math
\frac{dx}{dt} = qy - xy + x(1-x)
```

```math
\frac{dy}{dt} = -qy - xy + fz
```

```math
\frac{dz}{dt} = x-z
```

Here, `x`, `y` and `z` are the three model state variables. The plots associate `x` with HBrO₂ and `y` with Br⁻; `z` denotes the third state variable. The parameters `q` and `f` control coupling between the equations. These are the equations used in this educational simulation, rather than a quantitatively calibrated model of a particular experiment.

Because the system has **three state variables**, an `x`–`y` plot displays only a **projection** of the trajectory. The three-dimensional plot shows the complete path in the model's state space.

## Numerical methods and visualisation

- Integrates the three coupled nonlinear ODEs using `scipy.integrate.solve_ivp` with its adaptive RK45 method.
- Records the solution at 10,000 equally spaced output times. These are **output samples**, not fixed solver time steps.
- Checks the solver's success status and raises an error if integration fails.
- Displays **three time-series panels** for `x(t)`, `y(t)` and `z(t)` to examine their evolution and relative timescales.
- Plots a two-dimensional `x`–`y` phase-space projection and a full three-dimensional `(x, y, z)` trajectory.
- Marks the initial state in both phase-space plots.

## Project file and inputs

### `bz.py` — numerical simulation and visualisation

The script runs using settings defined near the top of the file; it **does not prompt for input**. Edit these values to investigate different behaviours:

```python
Q = 0.002
F = 1.2
X0 = [0.1, 0.2, 0.3]
T_SPAN = (0.0, 100.0)
NUM_OUTPUT_POINTS = 10000
```

| Setting | Meaning | Default |
| --- | --- | --- |
| `Q` | Coefficient coupling `y` to the `x` and `y` equations (`q` in the model) | `0.002` |
| `F` | Coefficient scaling the `z` contribution to `dy/dt` (`f` in the model) | `1.2` |
| `X0` | Initial conditions `[x(0), y(0), z(0)]` | `[0.1, 0.2, 0.3]` |
| `T_SPAN` | Simulation interval `(start_time, end_time)` | `(0.0, 100.0)` |
| `NUM_OUTPUT_POINTS` | Number of recorded output times | `10000` |

For example, setting `X0 = [0.2, 0.1, 0.2]` starts the simulation at a different point in state space. Changing `Q` or `F` changes the differential equations. The code then recomputes all three figures when rerun.

The default solver is adaptive RK45. The array constructed from `NUM_OUTPUT_POINTS` controls where SciPy **returns** the solution, not the solver's internal step size.

## Installation and usage

Requires Python and the following libraries:

```bash
pip install numpy scipy matplotlib
```

Run from the project directory:

```bash
python bz.py
```

The script displays three figures:

1. **Time-series plots:** `x(t)`, `y(t)` and `z(t)` in separate panels.
2. **2D phase-space projection:** the trajectory in the `x`–`y` plane, with the initial state marked.
3. **3D phase-space trajectory:** the path through `(x, y, z)` space, with the initial state marked.

Displaying interactive plots requires a graphical Matplotlib backend. To save a figure manually, use the Matplotlib window's save control.

## Learning outcomes

This exercise provided practice with numerical integration of coupled nonlinear ODEs, interpreting time evolution and trajectories in multidimensional phase space, exploring parameter and initial-condition dependence, and presenting simulation results using NumPy, SciPy and Matplotlib.

## Scope and limitations

This is an exploratory mathematical simulation, **not a quantitative prediction** of a specific BZ-reaction experiment. It plots one trajectory for the chosen parameters and initial state. It does not calculate equilibria, Jacobian eigenvalues, stability classifications or bifurcation diagrams. These are possible extensions.
