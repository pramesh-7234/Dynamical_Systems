import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.integrate import solve_ivp
import pandas as pd
from tabulate import tabulate
import math
import ast

params = input('Enter α, β, δ, γ in the form of a list, in the given order: ')
params = ast.literal_eval(params)

"""
x(t): Prey Population ; y(t) = Predator Population

alpha: Rate of growth of x
beta: Rate of decay of x
delta: Rate of growth of y
gamma: Rate of decay of y

dxdt = alpha * x - beta * x* y
dydt = delta * x * y - gamma * y
"""
alpha,beta,delta,gamma=params[0],params[1],params[2],params[3]
# Function returning derivative of x and y wrt time t
def lv_rhs(t,z):
    x,y=z
    dxdt=alpha*x - beta*x*y
    dydt=delta*x*y - gamma*y
    return [dxdt,dydt]

# Initial conditions determine where the system starts from
initial_conditions=input('Enter initial conditions in form of array (eg: [[x1,y1],[x2,y2]]):  ')
initial_conditions=ast.literal_eval(initial_conditions)
initial_conditions=np.array(initial_conditions)
# Example starting points: [[5.0, 2.0],[10.0, 5.0],[2.0, 10.0],[15.0, 1.0],[0.5, 0.5]]

t_end=float(input('Enter time span in seconds (eg: 60 for 60 seconds): ')) # duration of simulation
t_span=(0.0,t_end)
t_eval=np.linspace(t_span[0],t_span[1],int(t_span[1]*20)) # timesteps over which solve_ivp will integrate

solutions = []
for ic in initial_conditions:
    sol=solve_ivp(lv_rhs,t_span,ic,t_eval=t_eval,rtol=1e-8,atol=1e-10)
    solutions.append(sol)

fig,axs=plt.subplots(1,2,figsize=(15,8))
# Plotting the time dependence of x and y
for sol,ic in zip(solutions,initial_conditions): 
    axs[0].plot(sol.t,sol.y[0],label=f'x(t); IC={ic}')
    axs[0].plot(sol.t,sol.y[1],label=f'y(t); IC={ic}',linestyle='--')
   
axs[0].set_xlabel('t')
axs[0].set_ylabel('Population')
axs[0].set_title('Lotka-Volterra: Prey: solid, Predator: dashed')
axs[0].legend(loc='upper right')
axs[0].grid()  

## Plotting the phase plane : trajectories for initial conditions, nullclines and vector field

# Defining the grid for vector field:
x_vals=np.linspace(0,30.0,20)
y_vals=np.linspace(0,30.0,20)
X,Y=np.meshgrid(x_vals,y_vals)
U=alpha*X-beta*X*Y # dxdt
V=delta*X*Y-gamma*Y # dydt

speed=np.sqrt(U**2 + V**2) # Calculating dirn of the arrow at that point
speed[speed==0]=1.0 # to avoid division by zero
axs[1].quiver(X,Y,U/speed,V/speed,angles='xy',scale=50,width=0.002,alpha=0.7) # dividing by speed to normalize

# plotting the trajectories
for sol,ic in zip(solutions,initial_conditions):
    axs[1].plot(sol.y[0],sol.y[1],label=f'IC={ic}')

# plotting the nullclines:

# y_nullcline: dy/dt=0 ==> y=alpha/beta
# x_nullcline: dx/dt=0 ==> x=gamma/delta
y_nullcline=alpha/beta
x_nullcline=gamma/delta
axs[1].axhline(y_nullcline,linestyle='--')
axs[1].axvline(x_nullcline,linestyle='--')

# Fixed points: 
fixed_points=[[0.0,0.0],[x_nullcline,y_nullcline]]
for (x,y) in fixed_points:
    plt.plot(x,y,marker='o',markersize=8)
    plt.annotate(f'({x:.2f},{y:.2f})', xy=(x,y), xytext=(x+0.5,y+0.5))

axs[1].set_xlim(-1, 30)
axs[1].set_ylim(-1, 30)
axs[1].set_xlabel("x (prey)")
axs[1].set_ylabel("y (predator)")
axs[1].set_title("Phase plane: trajectories, vector field and nullclines")
axs[1].legend(loc='upper right', fontsize='small')
axs[1].grid(True)
plt.show()

# Compute Jacobian and eigenvalues at fixed points
def jacobian(x, y):
    return np.array([[alpha - beta*y, -beta*x],
                     [delta*y, delta*x - gamma]])

j_fp1 = jacobian(0.0, 0.0)
j_fp2 = jacobian(x_nullcline, y_nullcline)

# Compute eigenvalues 3 times each (to match user's preference for repeating calculations)
eigvals_fp1_runs = [la.eigvals(j_fp1) for _ in range(3)]
eigvals_fp2_runs = [la.eigvals(j_fp2) for _ in range(3)]

# Prepare a summary table
rows = []
rows.append({
    "Fixed point": "(0,0)",
    "Jacobian": str(j_fp1.tolist()),
    "Eigenvalues (runs)": "; ".join([str(np.round(ev,6)) for ev in eigvals_fp1_runs])
})
rows.append({
    "Fixed point": f"({x_nullcline:.3f}, {y_nullcline:.3f})",
    "Jacobian": str(j_fp2.tolist()),
    "Eigenvalues (runs)": "; ".join([str(np.round(ev,6)) for ev in eigvals_fp2_runs])
})

df_summary = pd.DataFrame(rows)

# Display summary to user via a neat dataframe
print("\nLotka–Volterra fixed points and eigenvalues (repeated calc):")
print(tabulate(df_summary, headers='keys', tablefmt='psql'))


# Also print numeric eigenvalues for clarity
print("Fixed point (0,0) Jacobian:\n", j_fp1)
print("Eigenvalues (3 runs):")
for i, ev in enumerate(eigvals_fp1_runs, 1):
    print(f" Run {i}: {ev}")

print("\nFixed point (x*, y*) = (gamma/delta, alpha/beta) Jacobian:\n", j_fp2)
print("Eigenvalues (3 runs):")
for i, ev in enumerate(eigvals_fp2_runs, 1):
    print(f" Run {i}: {ev}")

