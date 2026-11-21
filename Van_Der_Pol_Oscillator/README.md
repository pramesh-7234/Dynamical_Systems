# Van der Pol Oscillator

This project investigates the Van der Pol oscillator as an example of nonlinear, self-sustained oscillations. It compares the nonlinear system with its linearisation near equilibrium, visualises phase-plane trajectories, and provides a PyQt5 interface for experimenting with the oscillator parameter and initial conditions.

## Mathematical model

The dimensionless Van der Pol equation is

$$
\frac{d^2x}{dt^2}-\mu(1-x^2)\frac{dx}{dt}+x=0,
$$

where $\mu$ controls the nonlinear damping. Writing $v=dx/dt$ gives

$$
\frac{dx}{dt}=v, \qquad
\frac{dv}{dt}=\mu(1-x^2)v-x.
$$

The equilibrium is $(x,v)=(0,0)$, with Jacobian

```math
J(0,0) =
\begin{pmatrix}
0 & 1 \\
-1 & \mu
\end{pmatrix}
```

The Jacobian defines a local linear model near the origin. For $\mu>0$, the nonlinear oscillator can display a stable limit cycle, which cannot be represented by its local linearisation alone.

## Numerical methods and analysis

- Solves the nonlinear and linearised ODEs with SciPy's `solve_ivp` (default adaptive RK45 method).
- Computes the Jacobian using JAX automatic differentiation (`jacfwd`) and its eigenvalues and eigenvectors using SciPy.
- Compares nonlinear and linearised time series and phase-plane trajectories.
- Draws a vector field alongside the nonlinear phase portrait.
- Uses a PyQt5 interface to vary the nonlinear damping, initial conditions, and simulation duration.

The plots show how the linear approximation differs from the nonlinear dynamics as trajectories move away from the equilibrium. Nullclines are **not** drawn in the supplied scripts.

## Files and inputs

### `VDP.py` — standalone simulation

The script contains the following editable values:

```python
mu = 0.5            # Nonlinear damping parameter
t_span = (0, 40)    # Integration interval
x0 = [0.3, 0.0]    # Initial [position, velocity]
```

Modify these assignments in the file to run different cases. The script does **not** prompt for terminal input. It generates nonlinear/linearised time series and phase portraits, and prints the Jacobian, eigenvalues, and eigenvectors.

### `VDP_GUI.py` — interactive simulator

Enter or adjust the following settings:

| Input | Meaning | Default |
| --- | --- | --- |
| `mu` | Nonlinear damping coefficient $\mu$ | `0.5` |
| Initial condition | `x(0), v(0)` as comma-separated numbers | `0.3, 0.0` |
| Simulation end time | Slider for final time (5–200) | `40` |

The initial-condition field expects a pair such as `0.3, 0.0` **without brackets**. Click **Run Simulation** to display four plots: nonlinear time series, nonlinear phase portrait and vector field, linearised time series, and linearised phase portrait. The side panel displays the Jacobian and its eigenvalues/eigenvectors.

## Installation and usage

Requires NumPy, SciPy, Matplotlib, JAX, and PyQt5:

```bash
pip install numpy scipy matplotlib jax PyQt5
```

Run either version from the project directory:

```bash
python VDP.py
python VDP_GUI.py
```

The GUI needs a desktop environment with Qt support.

## Learning outcomes

This project provided experience with nonlinear ODE simulation, phase portraits and vector fields, automatic Jacobian calculation, eigenvalue-based local stability analysis, comparing nonlinear dynamics with linear approximations, and interactive parameter exploration.
