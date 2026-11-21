import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eig, expm
import jax.numpy as jnp
from jax import jacfwd

# Damped oscillator parameter
zeta = 0.1   # damping ratio (0 = undamped, >1 = overdamped)

def damped_oscillator(t, X):
    x1, x2 = X
    dx1dt = x2
    dx2dt = -x1 - 2*zeta*x2
    return jnp.array([dx1dt, dx2dt])

# Time span and initial condition
t_span = (0, 50)
t_eval = np.linspace(*t_span, 2000)
x0 = [1.0, 1.0]

# Solve numerically
sol_numerical = solve_ivp(damped_oscillator, t_span, x0, t_eval=t_eval)

#---------------------------------------------------------------------------------------
# Calculating the Jacobian and then plotting the analytical solution
# --------------------------------------------------------------------------------------

j_fun=jacfwd(lambda X: damped_oscillator(0,X))
x_eq=jnp.array([0.0,0.0])
jacobian=j_fun(x_eq)
vals,vecs=eig(jacobian)
vecs_real=np.real(vecs)
vecs_real=vecs_real/np.linalg.norm(vecs_real)

sol_analytic = np.column_stack([expm(jacobian * tt) @ x0 for tt in t_eval])

fig,axs=plt.subplots(2,2,figsize=(15,12))

# Plot time series
axs[0][0].plot(sol_numerical.t, sol_numerical.y[0], label="x1 (position)")
axs[0][0].plot(sol_numerical.t, sol_numerical.y[1], label="x2 (velocity)")

axs[0][1].plot(t_eval, sol_analytic[0,:], label='x1 (position)', linestyle='--')
axs[0][1].plot(t_eval, sol_analytic[1,:], label='x2 (velocity)', linestyle='--')
axs[0][0].set_xlabel('t'); axs[0][0].set_ylabel('Values')
axs[0][1].set_xlabel('t'); axs[0][1].set_ylabel('Values')
axs[0][0].legend(); axs[0][1].legend()
axs[0][0].grid(); axs[0][1].grid()
axs[0][0].set_title('Numerical Solution'); axs[0][1].set_title('Analytic Solution')

# Plot phase plane
x_vals = np.linspace(-1.5, 1.5, 20)
y_vals = np.linspace(-1.5, 1.5, 20)
Xg, Yg = np.meshgrid(x_vals, y_vals)
Ug = Yg
Vg = -Xg - 2*zeta*Yg
speed = np.sqrt(Ug**2 + Vg**2); speed[speed==0]=1.0
axs[1][0].quiver(Xg, Yg, Ug/speed, Vg/speed, angles='xy', scale=25, alpha=0.6)
axs[1][1].quiver(Xg, Yg, Ug/speed, Vg/speed, angles='xy', scale=25, alpha=0.6)

axs[1][0].plot(sol_numerical.y[0], sol_numerical.y[1], color='red')
axs[1][0].set_xlabel('x1 (position)'); axs[1][0].set_ylabel('x2 (velocity)')
axs[1][0].set_title('Phase plane trajectory (Numerical)')
axs[1][0].grid(True)

axs[1][1].plot(sol_analytic[0,:], sol_analytic[1,:], color='red')
axs[1][1].set_xlabel('x1 (position)'); axs[1][1].set_ylabel('x2 (velocity)')
axs[1][1].set_title('Phase plane trajectory (Analytic)')
axs[1][1].grid(True)

fig.suptitle(f'Damped harmonic oscillator (ζ={zeta})')
plt.show()

print(f'Jacobian: \n{jacobian}')
print(f'Eigenvalues: \n{vals}')
print(f'Eigenvectors: \n{vecs_real}')