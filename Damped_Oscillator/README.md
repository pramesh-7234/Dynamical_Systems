# Damped Harmonic Oscillator

This project investigates the dynamics of a damped harmonic oscillator through numerical integration, analytical solutions, and phase-plane analysis. It was developed as part of my independent study of dynamical systems and includes a Python script and an interactive PyQt5 simulator.

## Mathematical model

The dimensionless damped oscillator satisfies

$$
\frac{d^2x}{dt^2}+2\zeta\frac{dx}{dt}+x=0,
$$

where $x$ is displacement and $\zeta$ is the damping ratio. Introducing velocity $v=dx/dt$ gives the first-order system

$$
\frac{dx}{dt}=v, \qquad \frac{dv}{dt}=-x-2\zeta v.
$$

The equilibrium is $(x,v)=(0,0)$. Its Jacobian is

```math
J =
\begin{pmatrix}
0 & 1 \\
-1 & -2\zeta
\end{pmatrix}
```

The damping ratio determines the behaviour: $\zeta=0$ is undamped, $0<\zeta<1$ is underdamped, $\zeta=1$ is critically damped, and $\zeta>1$ is overdamped. Because the model is linear, its exact solution can also be computed using the matrix exponential:

$$
\mathbf{X}(t)=e^{Jt}\mathbf{X}(0), \qquad \mathbf{X}=(x,v)^T.
$$

## Numerical methods and analysis

- Integrates the system using SciPy's `solve_ivp` (default adaptive RK45 solver).
- Computes the analytical solution using `scipy.linalg.expm` for comparison with numerical integration.
- Uses JAX automatic differentiation (`jacfwd`) to calculate the Jacobian, followed by eigenvalue and eigenvector analysis.
- Plots displacement and velocity against time, phase-plane trajectories, and vector fields with Matplotlib.
- Provides a PyQt5 interface for testing different damping ratios and initial conditions.

## Files and inputs

### `Damped_Oscillator.py` — standalone simulation

The script runs with values defined near the beginning of the file:

```python
zeta = 0.1          # Damping ratio
t_span = (0, 50)    # Integration interval
x0 = [1.0, 1.0]    # Initial [displacement, velocity]
```

Edit these assignments to explore different cases; this script does **not** request terminal input. It generates four plots comparing numerical and analytical time series and phase-plane trajectories, then prints the Jacobian, eigenvalues, and eigenvectors.

### `Damped_Oscillator_GUI.py` — interactive simulator

Enter the following values in the application:

| Input | Meaning | Default |
| --- | --- | --- |
| `ζ` | Damping ratio | `0.1` |
| `Initial x1` | Initial displacement $x(0)$ | `1.0` |
| `Initial x2` | Initial velocity $v(0)$ | `1.0` |
| Simulation end time | Slider for final time (5–100) | `50` |

All three text inputs accept a single numerical value. Click **Run Simulation** to refresh the numerical/analytical comparisons and phase portraits. The interface also displays the Jacobian and its eigenvalues/eigenvectors.

## Installation and usage

Requires Python with NumPy, SciPy, Matplotlib, JAX, and PyQt5. For a standard CPU installation:

```bash
pip install numpy scipy matplotlib jax PyQt5
```

Run either version from the project directory:

```bash
python Damped_Oscillator.py
python Damped_Oscillator_GUI.py
```

The GUI needs a desktop environment with Qt support.

## Learning outcomes

This exercise developed experience in numerical ODE integration, matrix-exponential solutions, phase-plane analysis, Jacobian/eigenvalue calculations, comparison of numerical and analytical solutions, and interactive scientific visualisation.
