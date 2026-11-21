# Lotka–Volterra Predator–Prey Model

This project studies the Lotka–Volterra predator–prey system through numerical integration, phase-plane analysis, and local equilibrium analysis. Developed during my independent study of dynamical systems, it includes a command-line simulation and an interactive PyQt5 application for exploring how parameters and initial conditions affect the evolution of the system.

## Mathematical model

The interacting populations are described by two coupled nonlinear ordinary differential equations:

$$
\frac{dx}{dt} = \alpha x - \beta xy
$$

$$
\frac{dy}{dt} = \delta xy - \gamma y
$$

Here, $x(t)$ and $y(t)$ are the prey and predator populations, respectively. The four model parameters are:

| Parameter | Meaning |
| --- | --- |
| $\alpha$ (`alpha`) | Prey growth rate in the absence of predators |
| $\beta$ (`beta`) | Predation coefficient, determining prey loss through encounters |
| $\delta$ (`delta`) | Predator growth coefficient, representing gains from encounters with prey |
| $\gamma$ (`gamma`) | Predator mortality rate in the absence of prey |

For positive parameter values, the two equilibrium points are:

$$
E_1=(0,0), \qquad E_2=\left(\frac{\gamma}{\delta},\frac{\alpha}{\beta}\right)
$$

The Jacobian matrix is calculated at each equilibrium to examine its local behaviour:

$$
J(x,y)=
\begin{pmatrix}
\alpha-\beta y & -\beta x \\
\delta y & \delta x-\gamma
\end{pmatrix}
$$

## Numerical methods and analysis

- Integrates the coupled ODEs using `scipy.integrate.solve_ivp` with its default adaptive RK45 method.
- Compares prey and predator population histories for different initial conditions.
- Plots trajectories and direction fields in the phase plane.
- Displays the non-axis nullclines $y=\alpha/\beta$ and $x=\gamma/\delta$, along with the equilibrium points.
- Calculates Jacobian matrices and eigenvalues at the equilibria for local stability analysis.
- Provides a GUI to explore the effect of changing parameters and initial conditions without editing the code.

## Project files

### `LV_Model.py` — Command-line simulation

Runs the model from the terminal. The user is prompted for **three inputs**, in this order:

1. **Model parameters:** Enter a Python list of four numbers in the order `[alpha, beta, delta, gamma]`, for example `[1.1, 0.4, 0.1, 0.4]`. The order matters because each value controls a different term in the equations.
2. **Initial conditions:** Enter a nested list of population pairs, for example `[[10, 5], [5, 2]]`. Each pair is `[initial prey population, initial predator population]`. The script solves the equations separately for every pair and compares the resulting trajectories on the same plots. To test only one starting point, use a nested list such as `[[10, 5]]`.
3. **Simulation duration:** Enter a positive number such as `60`, representing the end of the simulated time interval, which starts at $t=0$. Time units should be consistent with the rates chosen for the parameters.

The script produces population time-series and phase-plane plots, then prints the equilibrium points, their Jacobian matrices, and the corresponding eigenvalues. The phase-plane vector field in this version is drawn on a fixed grid from 0 to 30 for each population.

### `LV_GUI.py` — Interactive simulator

Provides the same underlying model in a PyQt5 window. Instead of responding to terminal prompts, the user can enter:

- **Four separate parameter values:** `alpha`, `beta`, `delta`, and `gamma`. The starting values are `1.1`, `0.4`, `0.1`, and `0.4`, respectively.
- **Initial population pairs:** A nested list in the **Initial Conditions (x,y)** field. The default `[[10,5],[5,2]]` represents two simulations: one beginning with 10 prey and 5 predators, and another beginning with 5 prey and 2 predators. Additional pairs can be entered using the same format.
- **Simulation time:** A slider selecting the end time from 10 to 200, with a default of 60.

Select **Run Simulation** to update the population time series, phase-plane trajectories, vector field, nullclines, and equilibrium markers. A text panel displays the Jacobian matrices and eigenvalues at both equilibria. The phase-plane plotting range adjusts to the simulated trajectories.

For both versions, **positive model parameters** and **non-negative initial populations** are appropriate for the standard biological interpretation. The input formats above should be followed exactly.

## Requirements

Python 3 and the following packages:

- **NumPy** — arrays, numerical operations, and eigenvalues
- **SciPy** — numerical ODE integration
- **Matplotlib** — time-series and phase-plane visualisation
- **pandas** and **tabulate** — tabular output in `LV_Model.py`
- **PyQt5** — graphical interface in `LV_GUI.py`

Install them with:

```bash
pip install numpy scipy matplotlib pandas tabulate PyQt5
```

## Running the programs

For the command-line version:

```bash
python LV_Model.py
```

Example sequence of responses to its prompts:

```text
[1.1, 0.4, 0.1, 0.4]
[[10, 5], [5, 2]]
60
```

For the graphical version:

```bash
python LV_GUI.py
```

## Learning focus

This project provided experience in implementing and numerically solving nonlinear ODE systems, interpreting trajectories and phase portraits, identifying nullclines and equilibria, and using Jacobian eigenvalues for local analysis. Building the graphical interface also provided practice in creating interactive tools for exploring mathematical models.
