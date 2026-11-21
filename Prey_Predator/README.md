# Lotka–Volterra Predator–Prey Model

This project explores the Lotka–Volterra predator–prey model through numerical simulation and phase-plane analysis. 
Developed as part of my independent study of dynamical systems, it includes a command-line script and an interactive PyQt5 application for investigating how model parameters and initial conditions affect population dynamics.

## Mathematical Model

The system is described by two coupled nonlinear ordinary differential equations:

$$
\frac{dx}{dt}=\alpha x-\beta xy, \qquad
\frac{dy}{dt}=\delta xy-\gamma y
$$

Here, \(x\) and \(y\) represent prey and predator populations. The parameters \(\alpha\)\, (\beta\), \(\delta\) and \(\gamma\) represent prey growth, predation, predator growth through predation and predator mortality.

For positive parameter values, the equilibrium points are \((0,0)\) and \((\gamma/\delta,\alpha/\beta)\).

## Numerical Methods and Analysis

The project includes:

- Numerical integration of the coupled ODEs using SciPy's `solve_ivp` with its default adaptive RK45 method.
- Analysis of population evolution for different initial conditions.
- Phase-plane visualisation using trajectories and vector fields.
- Calculation and visualisation of nullclines and equilibrium points.
- Jacobian matrix and eigenvalue calculations for local equilibrium analysis.
- An interactive graphical interface for exploring different model parameters.

## Implementation

**Python libraries:** NumPy, SciPy, Matplotlib, pandas, tabulate and PyQt5.

**Files:**

- `LV_Model.py` — Command-line simulation with population plots, phase portraits and equilibrium analysis.
- `LV_GUI.py` — Interactive simulator with adjustable parameters, initial conditions, simulation duration and graphical output.

## Installation and Usage

Install the required libraries:

`pip install numpy scipy matplotlib pandas tabulate PyQt5`

Run the command-line simulation:

`python LV_Model.py`

The program accepts the four model parameters, multiple initial conditions and the simulation duration.

Launch the graphical interface:

`python LV_GUI.py`

The GUI provides adjustable simulation settings, time-series and phase-plane plots, and numerical results for the equilibrium points.

## Learning Outcomes

This project provided practical experience in solving nonlinear ODE systems, constructing phase portraits, analysing nullclines and equilibrium points, and using Jacobian eigenvalues to examine local stability. Developing the GUI also provided experience in building interactive tools for exploring mathematical models.
