import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eig
import jax.numpy as jnp
from jax import jacfwd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSlider, QTextEdit, QGroupBox
)
from PyQt5.QtCore import Qt

# ------------------------------
# FitzHugh–Nagumo Simulation
# ------------------------------
def FN_rhs(t, X, a, b, e, Iext):
    v, w = X
    dvdt = v - (v**3)/3 - w + Iext
    dwdt = e*(v + a - b*w)
    return jnp.array([dvdt, dwdt])

def simulate_FHN(a, b, e, Iext, x0_list, t_end):
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, 6000)
    solutions = [solve_ivp(lambda t, X: FN_rhs(t, X, a, b, e, Iext),
                           t_span, x0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
                 for x0 in x0_list]

    # 3D surfaces
    v_vals = np.linspace(-3, 3, 50)
    w_vals = np.linspace(-3, 3, 50)
    V, W = np.meshgrid(v_vals, w_vals)
    DV = np.zeros_like(V)
    DW = np.zeros_like(W)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            dv, dw = FN_rhs(0, [V[i,j], W[i,j]], a, b, e, Iext)
            DV[i,j] = dv
            DW[i,j] = dw

    # Phase plane limits
    v_min = min(sol.y[0].min() for sol in solutions)
    v_max = max(sol.y[0].max() for sol in solutions)
    w_min = min(sol.y[1].min() for sol in solutions)
    w_max = max(sol.y[1].max() for sol in solutions)

    # Fixed point (real root)
    p = [-1/3, 0, 1 - 1/b, Iext - a/b]
    v_roots = np.roots(p)
    v_fp = v_roots[np.isreal(v_roots)].real[0]
    w_fp = (v_fp + a)/b

    # Jacobian at fixed point
    j_fun = jacfwd(lambda X: FN_rhs(0,X,a,b,e,Iext))
    x_eq = jnp.array([v_fp, w_fp])
    jacobian = j_fun(x_eq)
    vals, vecs = eig(jacobian)
    vecs_real = np.real(vecs)
    v_norm = vecs_real[:,0]/np.linalg.norm(vecs_real[:,0])

    # Linearized solutions
    def linear_rhs(t, x):
        delta = x - np.array([v_fp, w_fp])
        return jacobian @ delta
    linear_solutions = [solve_ivp(linear_rhs, t_span, x0, t_eval=t_eval,
                                  rtol=1e-8, atol=1e-8)
                        for x0 in x0_list]

    return solutions, linear_solutions, V, W, DV, DW, (v_fp, w_fp), jacobian, vals, vecs_real, v_norm, (v_min, v_max, w_min, w_max), t_eval

# ------------------------------
# PyQt5 GUI
# ------------------------------
class FHNApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FitzHugh–Nagumo Simulator")
        self.setGeometry(50,50,1700,900)

        layout = QHBoxLayout(self)

        # Left panel: Controls + Output
        control_layout = QVBoxLayout()

        # Parameter inputs
        param_box = QGroupBox("Parameters")
        param_layout = QVBoxLayout()
        self.a_input = QLineEdit("0.7")
        self.b_input = QLineEdit("0.8")
        self.e_input = QLineEdit("0.08")
        self.I_input = QLineEdit("0.5")
        param_layout.addWidget(QLabel("a:")); param_layout.addWidget(self.a_input)
        param_layout.addWidget(QLabel("b:")); param_layout.addWidget(self.b_input)
        param_layout.addWidget(QLabel("e:")); param_layout.addWidget(self.e_input)
        param_layout.addWidget(QLabel("Iext:")); param_layout.addWidget(self.I_input)
        param_box.setLayout(param_layout)
        control_layout.addWidget(param_box)

        # Initial conditions
        ic_box = QGroupBox("Initial Conditions (v,w)")
        ic_layout = QVBoxLayout()
        self.ic1_input = QLineEdit("-1.5, -1.0")
        self.ic2_input = QLineEdit("0.5, 0.0")
        ic_layout.addWidget(QLabel("IC1:")); ic_layout.addWidget(self.ic1_input)
        ic_layout.addWidget(QLabel("IC2:")); ic_layout.addWidget(self.ic2_input)
        ic_box.setLayout(ic_layout)
        control_layout.addWidget(ic_box)

        # Slider for runtime
        slider_box = QGroupBox("Simulation runtime")
        slider_layout = QVBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(50)
        self.slider.setMaximum(500)
        self.slider.setValue(200)
        self.slider.valueChanged.connect(self.update_slider_label)
        self.slider_label = QLabel("Simulation End Time: 200")
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.slider_label)
        slider_box.setLayout(slider_layout)
        control_layout.addWidget(slider_box)

        # Run button
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        control_layout.addWidget(self.run_button)

        # Output text
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        control_layout.addWidget(self.output_text)

        layout.addLayout(control_layout,1)

        # Right panel: Plots
        self.fig = plt.figure(figsize=(14,10))

        # 3D surfaces
        self.ax0 = self.fig.add_subplot(3,2,1, projection='3d')
        self.ax1 = self.fig.add_subplot(3,2,2, projection='3d')

        # 2D plots
        self.ax2 = self.fig.add_subplot(3,2,3)
        self.ax3 = self.fig.add_subplot(3,2,4)
        self.ax4 = self.fig.add_subplot(3,2,5)
        self.ax5 = self.fig.add_subplot(3,2,6)

        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas,4)

    def update_slider_label(self):
        self.slider_label.setText(f"Simulation End Time: {self.slider.value()}")

    def run_simulation(self):
        try:
            a = float(self.a_input.text())
            b = float(self.b_input.text())
            e = float(self.e_input.text())
            Iext = float(self.I_input.text())
            t_end = self.slider.value()
            x0_list = []
            for ic_str in [self.ic1_input.text(), self.ic2_input.text()]:
                v0,w0 = map(float, ic_str.split(","))
                x0_list.append([v0,w0])

            (solutions, linear_solutions, V, W, DV, DW, fp,
             jacobian, vals, vecs_real, v_norm, limits, t_eval) = simulate_FHN(a,b,e,Iext,x0_list,t_end)

            v_min, v_max, w_min, w_max = limits

            # Clear previous plots
            for ax in [self.ax0, self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
                ax.clear()

            # 3D dv/dt surface
            self.ax0.plot_surface(V,W,DV,cmap='viridis', alpha=0.8)
            self.ax0.set_xlabel('v'); self.ax0.set_ylabel('w'); self.ax0.set_zlabel('dv/dt')
            self.ax0.set_title('Surface of dv/dt')

            # 3D dw/dt surface
            self.ax1.plot_surface(V,W,DW,cmap='plasma', alpha=0.8)
            self.ax1.set_xlabel('v'); self.ax1.set_ylabel('w'); self.ax1.set_zlabel('dw/dt')
            self.ax1.set_title('Surface of dw/dt')

            # Time evolution (nonlinear)
            for sol, ic in zip(solutions, x0_list):
                self.ax2.plot(sol.t, sol.y[0], label=f'v IC={ic}')
                self.ax2.plot(sol.t, sol.y[1], '--', label=f'w IC={ic}')
            self.ax2.set_xlabel('Time'); self.ax2.set_ylabel('Value')
            self.ax2.set_title('Time evolution (Nonlinear)')
            self.ax2.grid(True)
            self.ax2.legend()

            # Phase plane + trajectories
            v_grid = np.linspace(v_min,v_max,20)
            w_grid = np.linspace(w_min,w_max,20)
            Vf,Wf = np.meshgrid(v_grid,w_grid)
            DVf = Vf - (Vf**3)/3 - Wf + Iext
            DWf = e*(Vf + a - b*Wf)
            speed = np.sqrt(DVf**2 + DWf**2)
            speed[speed==0]=1.0
            self.ax3.quiver(Vf,Wf,DVf/speed,DWf/speed,angles='xy',scale=30,alpha=0.5)
            for sol, ic in zip(solutions, x0_list):
                self.ax3.plot(sol.y[0], sol.y[1], label=f'IC={ic}')
            # Nullclines
            v_vals_null = np.linspace(v_min,v_max,2000)
            w_null1 = v_vals_null - (v_vals_null**3)/3 + Iext
            w_null2 = (v_vals_null + a)/b
            mask1 = (w_min < w_null1) & (w_null1 < w_max)
            mask2 = (w_min < w_null2) & (w_null2 < w_max)
            self.ax3.plot(v_vals_null[mask1], w_null1[mask1],'--',label='Nullcline1')
            self.ax3.plot(v_vals_null[mask2], w_null2[mask2],'--',label='Nullcline2')
            self.ax3.plot(fp[0],fp[1], marker='o', color='black')
            self.ax3.annotate(f'({fp[0]:.2f},{fp[1]:.2f})',(fp[0]+0.05,fp[1]-0.05))
            self.ax3.set_xlabel('v'); self.ax3.set_ylabel('w')
            self.ax3.set_title('Phase plane & trajectories')
            self.ax3.grid(True)
            self.ax3.legend()

            # Linear evolution (time)
            for lin_sol, ic in zip(linear_solutions, x0_list):
                self.ax4.plot(lin_sol.t, lin_sol.y[0], label=f'v Linear IC={ic}')
                self.ax4.plot(lin_sol.t, lin_sol.y[1], label=f'w Linear IC={ic}')
            self.ax4.set_xlabel('Time'); self.ax4.set_ylabel('Value')
            self.ax4.set_title('Time evolution (Linear)')
            self.ax4.grid(True)
            self.ax4.legend()

            # Linearized phase plane
            for lin_sol, ic in zip(linear_solutions, x0_list):
                self.ax5.plot(lin_sol.y[0], lin_sol.y[1], label=f'IC={ic}')
            self.ax5.set_xlabel('v'); self.ax5.set_ylabel('w')
            self.ax5.set_title('Linearized FHN')
            self.ax5.grid(True)
            self.ax5.legend()

            self.canvas.draw()

            # Update side panel
            output = f"Fixed point: v_fp= {fp[0]:.4f}, w_fp= {fp[1]:.4f}\n\n"
            output += f"Jacobian:\n{np.round(jacobian,4)}\n\n"
            output += f"Eigenvalues:\n{np.round(vals,4)}\n\n"
            output += f"Eigenvectors (columns):\n{np.round(vecs_real,4)}\n\n"
            output += f"Normalized first eigenvector: {np.round(v_norm,4)}"
            self.output_text.setText(output)

        except Exception as ex:
            self.output_text.setText(f"Error: {ex}")

# ------------------------------
# Run application
# ------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FHNApp()
    window.show()
    sys.exit(app.exec_())
