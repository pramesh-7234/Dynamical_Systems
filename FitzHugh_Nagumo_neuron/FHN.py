import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eig
import jax.numpy as jnp
from jax import jacfwd

# FitzHugh–Nagumo model parameters
Iext = 0.5
a = 0.7
b = 0.8
e = 0.08

# Non Linear Model
def FN_rhs(t, X):
    v, w = X
    dvdt = v - (v**3)/3 - w + Iext
    dwdt = e*(v + a - b*w)
    return jnp.array([dvdt, dwdt])

# Time evolution setup
t_span = (0.0, 200.0)
t_eval = np.linspace(t_span[0], t_span[1], 6000)
initial_conditions = [[-1.5, -1.0],[0.5, 0.0]]

#Numerical solutions
sol_numerical = [solve_ivp(FN_rhs, t_span, ic, t_eval=t_eval, rtol=1e-8, atol=1e-10)
             for ic in initial_conditions]



# 3D surface data
v_vals = np.linspace(-3, 3, 50)
w_vals = np.linspace(-3, 3, 50)
V, W = np.meshgrid(v_vals, w_vals)
DV = np.zeros_like(V)
DW = np.zeros_like(W)

for i in range(V.shape[0]):
    for j in range(V.shape[1]):
        dv, dw = FN_rhs(0,[V[i,j], W[i,j]])
        DV[i,j] = dv
        DW[i,j] = dw

# Determine phase plane limits
v_min = min(sol.y[0].min() for sol in sol_numerical)
v_max = max(sol.y[0].max() for sol in sol_numerical)
w_min = min(sol.y[1].min() for sol in sol_numerical)
w_max = max(sol.y[1].max() for sol in sol_numerical)

# Create 2x2 subplot grid
fig = plt.figure(figsize=(15,12))

# -------------------------------
# Top-left: dv/dt 3D surface
# -------------------------------
ax0 = fig.add_subplot(3,2,1, projection='3d')
ax0.plot_surface(V, W, DV, cmap='viridis', alpha=0.8)
ax0.set_xlabel('v'); ax0.set_ylabel('w'); ax0.set_zlabel('dv/dt')
ax0.set_title('Surface of dv/dt')

# -------------------------------
# Top-right: dw/dt 3D surface
# -------------------------------
ax1 = fig.add_subplot(3,2,2, projection='3d')
ax1.plot_surface(V, W, DW, cmap='plasma', alpha=0.8)
ax1.set_xlabel('v'); ax1.set_ylabel('w'); ax1.set_zlabel('dw/dt')
ax1.set_title('Surface of dw/dt')

# -------------------------------
# Bottom-left: Time evolution
# -------------------------------
ax2 = fig.add_subplot(3,2,3)
for sol, ic in zip(sol_numerical, initial_conditions):
    ax2.plot(sol.t, sol.y[0], label=f'v IC={ic}')
    ax2.plot(sol.t, sol.y[1], '--', label=f'w IC={ic}')
ax2.set_xlabel('Time'); ax2.set_ylabel('Value')
ax2.set_title('Time evolution of v and w')
ax2.grid(True)
ax2.legend()

# -------------------------------
# Bottom-right: Phase plane + trajectories
# -------------------------------
ax3 = fig.add_subplot(3,2,4)

v_grid = np.linspace(v_min, v_max, 20)
w_grid = np.linspace(w_min, w_max, 20)
Vf, Wf = np.meshgrid(v_grid, w_grid)

DVf = Vf - (Vf**3)/3 - Wf + Iext
DWf = e*(Vf + a - b*Wf)
speed = np.sqrt(DVf**2 + DWf**2)
speed[speed==0] = 1.0
ax3.quiver(Vf, Wf, DVf/speed, DWf/speed, angles='xy', scale=30, alpha=0.5)

for sol, ic in zip(sol_numerical, initial_conditions):
    ax3.plot(sol.y[0], sol.y[1], label=f'IC={ic}')

# Optional: nullclines
v_vals_null = np.linspace(v_min, v_max, 2000)
w_null1 = v_vals_null - (v_vals_null**3)/3 + Iext
w_null2 = (v_vals_null + a)/b
mask1 = (w_min < w_null1) & (w_null1 < w_max)
mask2 = (w_min < w_null2) & (w_null2 < w_max)
ax3.plot(v_vals_null[mask1], w_null1[mask1], '--', label='Nullcline 1')
ax3.plot(v_vals_null[mask2], w_null2[mask2], '--', label='Nullcline 2')

# Coefficients of cubic: -v^3/3 + v*(1-1/b) + (Iext - a/b) = 0
p = [-1/3, 0, 1 - 1/b, Iext - a/b]  # descending powers: v^3, v^2, v^1, v^0
v_roots = np.roots(p)

# Keep only real roots
v_fp = v_roots[np.isreal(v_roots)].real[0]  # or select the root you want
w_fp = (v_fp + a)/b

ax3.plot(v_fp,w_fp,marker='o',color='black')
ax3.annotate(f'({v_fp:.2f}, {w_fp:.2f})',(v_fp+0.1,w_fp-0.1))

ax3.set_xlabel('v'); ax3.set_ylabel('w')
ax3.set_title('Phase plane & trajectories')
ax3.grid(True)
ax3.legend()

j_fun=jacfwd(lambda X: FN_rhs(0,X))
x_eq=jnp.array([v_fp,w_fp])
jacobian=j_fun(x_eq)

def linear_rhs(t, x):
    delta = x - np.array([v_fp,w_fp])
    return jacobian @ delta

linear_solutions = [
    solve_ivp(linear_rhs, t_span, ic, t_eval=t_eval, rtol=1e-8, atol=1e-8)
    for ic in initial_conditions
]

ax4 = fig.add_subplot(3,2,5)
# Add the fixed point back when plotting
for lin_sol, ic in zip(linear_solutions, initial_conditions):
    ax4.plot(lin_sol.t, lin_sol.y[0], label=f'(v Linear IC={ic})')
    ax4.plot(lin_sol.t, lin_sol.y[1], label=f'(w Linear IC={ic})')

ax4.set_ylabel('Value'); ax4.set_xlabel('Time')
ax4.set_title('Time evolution of v and w (Linear)')
ax4.grid()
ax4.legend()

ax5 = fig.add_subplot(3,2,6)
# Add the fixed point back when plotting
for lin_sol, ic in zip(linear_solutions, initial_conditions):
    v_lin = lin_sol.y[0]
    w_lin = lin_sol.y[1] 
    ax5.plot(v_lin, w_lin, label=f'(IC={ic})')

ax5.set_ylabel('w')
ax5.set_xlabel('v')
ax5.set_title('Linearized FHN')
ax5.legend()
ax5.grid()


plt.tight_layout()
plt.show()
