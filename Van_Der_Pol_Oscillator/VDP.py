import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eig
import jax.numpy as jnp
from jax import jacfwd


# Van der Pol parameters
mu = 0.5

# Non Linear function
def vdp(t, X):
    x1, x2 = X
    dx1dt = x2
    dx2dt = mu * (1 - x1**2) * x2 - x1
    return jnp.array([dx1dt, dx2dt])

# Calculating the jacobian:
j_fun=jacfwd(lambda X: vdp(0,X))
x_eq=jnp.array([0.0,0.0])
jacobian=j_fun(x_eq)

# Linear Van Der Pol Model
def linear_vdp(t,X):
    return jacobian @ X

# Eigenvalues and Eigenvectors
vals,vecs=eig(jacobian)
vecs_real = np.real(vecs)
vecs_real_norm = vecs_real / np.linalg.norm(vecs_real)


# Time span and initial condition
t_span = (0, 40)
t_eval = np.linspace(*t_span, 2000)
x0 = [0.3, 0.0]

# Solve numerically
sol = solve_ivp(vdp, t_span, x0, t_eval=t_eval,rtol=1e-8,atol=1e-10)
linear_sol=solve_ivp(linear_vdp,t_span,x0,t_eval=t_eval,rtol=1e-8,atol=1e-8)

# Creating meshgrid for plotting quivers:
x_vals = np.linspace(sol.y[0].min(),sol.y[0].max(),20)
y_vals = np.linspace(sol.y[1].min(),sol.y[1].max(),20)
X,Y = np.meshgrid(x_vals,y_vals)
U = Y
V = mu * (1 - X**2) * Y - X
speed= np.sqrt(U**2 + V**2)
speed[speed==0.0]=1.0


#-------------------------------------------------------------------------------
# PLOTS 
#-------------------------------------------------------------------------------

fig,axs=plt.subplots(2,2,figsize=(10,10))
# Plot time series (non-linear)
axs[0][0].plot(sol.t, sol.y[0], label="x1 (position)")
axs[0][0].plot(sol.t, sol.y[1], label="x2 (velocity)", linestyle='--')
axs[0][0].set_xlabel('t'); axs[0][0].set_ylabel('Values')
axs[0][0].legend(); axs[0][0].set_title('Time variation of x1 and x2')
axs[0][0].grid(True)

axs[1][0].plot(linear_sol.t, linear_sol.y[0], label="x1 (position)")
axs[1][0].plot(linear_sol.t, linear_sol.y[1], label="x2 (velocity)", linestyle='--')
axs[1][0].set_xlabel('t'); axs[1][0].set_ylabel('Values')
axs[1][0].legend(); axs[1][0].set_title('Time variation of x1 and x2 (Linearized)')
axs[1][0].grid(True)

# Plot phase plane
axs[0][1].quiver(x_vals, y_vals, U/speed, V/speed, angles='xy', scale=40, alpha=0.5)
axs[0][1].plot(sol.y[0], sol.y[1], color='darkorange',label=f'IC={x0}')
axs[0][1].set_xlabel('x1'); axs[0][1].set_ylabel('x2')
axs[0][1].set_title('Phase plane trajectory')
axs[0][1].plot(0,0,marker='o',color='black')
axs[0][1].annotate('(0.0,0.0)',xy=(0,0),xytext=(0,0+0.2))
axs[0][1].grid(True)
axs[0][1].legend()

axs[1][1].plot(linear_sol.y[0], linear_sol.y[1], color='darkorange',label=f'IC={x0}')
axs[1][1].set_xlabel('x1'); axs[1][1].set_ylabel('x2')
axs[1][1].set_title('Phase plane trajectory')
axs[1][1].plot(0,0,marker='o',color='black')
axs[1][1].annotate('(0.0,0.0)',xy=(0,0),xytext=(0,0+0.2))
axs[1][1].grid(True)
axs[1][1].legend()

fig.suptitle(f'Van der Pol oscillator (μ={mu})')
plt.show()

# Nullclines not drawn as they become asymptotes


print(f'Jacobian: \n{jacobian}')
print(f'Eigenvalues: \n{vals}')
print(f'Eigenvectors (normalised): \n{vecs_real_norm}')