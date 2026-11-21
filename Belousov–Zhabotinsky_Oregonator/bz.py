"""Numerical exploration of a simplified Oregonator model of the BZ reaction.

Run ``python bz.py`` to display time-series and phase-space plots.
Edit the settings below to investigate other parameters and initial states.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# -----------------------------------------------------------------------------
# Simulation settings (edit these to explore different trajectories)
# -----------------------------------------------------------------------------
Q = 0.002                 # Coefficient coupling y to the x and y equations
F = 1.2                   # Coefficient coupling z to the y equation
X0 = [0.1, 0.2, 0.3]      # Initial state: [x(0), y(0), z(0)]
T_SPAN = (0.0, 100.0)     # Start and end times
NUM_OUTPUT_POINTS = 10000  # Output samples, not integration steps


def oregonator(t, state, q=Q, f=F):
    """Return derivatives for the simplified three-variable Oregonator."""
    x, y, z = state
    dxdt = q * y - x * y + x * (1.0 - x)
    dydt = -q * y - x * y + f * z
    dzdt = x - z
    return [dxdt, dydt, dzdt]


def main():
    t_eval = np.linspace(*T_SPAN, NUM_OUTPUT_POINTS)

    # RK45 is adaptive. t_eval specifies where the solution is recorded.
    sol = solve_ivp(oregonator, T_SPAN, X0, t_eval=t_eval, method="RK45")
    if not sol.success:
        raise RuntimeError(f"Numerical integration failed: {sol.message}")

    t = sol.t
    x, y, z = sol.y

    # 1. Time evolution: separate panels make it easy to compare timescales.
    fig_time, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, values, name, label in zip(
        axes,
        (x, y, z),
        ("x", "y", "z"),
        ("x (HBrO2)", "y (Br-)", "z (third state variable)"),
    ):
        ax.plot(t, values, label=name)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (model units)")
    fig_time.suptitle("Oregonator: evolution of the three state variables")
    fig_time.tight_layout()

    # 2. Two-dimensional projection of the three-dimensional trajectory.
    fig_xy, ax_xy = plt.subplots(figsize=(7, 5))
    ax_xy.plot(x, y, linewidth=1.2)
    ax_xy.scatter([x[0]], [y[0]], marker="o", label="Initial state")
    ax_xy.set_xlabel("x (HBrO2)")
    ax_xy.set_ylabel("y (Br-)")
    ax_xy.set_title("Oregonator: x-y phase-space projection")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend()
    fig_xy.tight_layout()

    # 3. Full three-dimensional phase-space trajectory.
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.plot(x, y, z, linewidth=1.2)
    ax_3d.scatter([x[0]], [y[0]], [z[0]], marker="o", label="Initial state")
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.set_title("Oregonator: 3D phase-space trajectory")
    ax_3d.legend()
    fig_3d.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
