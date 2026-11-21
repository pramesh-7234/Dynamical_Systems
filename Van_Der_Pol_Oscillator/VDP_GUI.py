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
# Van der Pol model (JAX for jacobian)
# ------------------------------
def vdp_jax(X, mu):
    x1, x2 = X
    dx1dt = x2
    dx2dt = mu * (1 - x1**2) * x2 - x1
    return jnp.array([dx1dt, dx2dt])

# pure-numpy wrapper for ODE solver
def vdp_np(t, X, mu):
    x1, x2 = X
    dx1dt = x2
    dx2dt = mu * (1 - x1**2) * x2 - x1
    return np.array([dx1dt, dx2dt])

# ------------------------------
# Simulation helper
# ------------------------------
def simulate_vdp(mu, x0, t_end):
    # compute Jacobian via JAX at equilibrium (0,0)
    j_fun = jacfwd(lambda X: vdp_jax(X, mu))
    x_eq = jnp.array([0.0, 0.0])
    jacobian_jax = j_fun(x_eq)               # jnp array
    jacobian = np.array(jacobian_jax)        # convert to numpy for eig

    # eigen-analysis
    vals, vecs = eig(jacobian)
    vecs_real = np.real(vecs)
    # normalize eigenvectors column-wise
    norms = np.linalg.norm(vecs_real, axis=0)
    # avoid division by zero
    norms[norms == 0] = 1.0
    vecs_real_norm = vecs_real / norms

    # linear system rhs using the Jacobian
    def linear_rhs(t, X):
        return jacobian.dot(X)

    # integrate nonlinear and linear systems
    t_span = (0.0, t_end)
    t_eval = np.linspace(0.0, t_end, max(200, int(t_end * 50)))

    sol_nonlinear = solve_ivp(lambda t, X: vdp_np(t, X, mu), t_span, x0, t_eval=t_eval,
                              rtol=1e-8, atol=1e-10)
    sol_linear = solve_ivp(linear_rhs, t_span, x0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

    return {
        "jacobian": jacobian,
        "eigvals": vals,
        "eigvecs": vecs_real,
        "eigvecs_norm": vecs_real_norm,
        "sol_nonlinear": sol_nonlinear,
        "sol_linear": sol_linear,
        "t_eval": t_eval
    }

# ------------------------------
# PyQt5 GUI
# ------------------------------
class VDPApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Van der Pol Simulator")
        self.setGeometry(50, 50, 1400, 900)

        main_layout = QHBoxLayout(self)

        # Left panel: inputs & outputs
        left_layout = QVBoxLayout()

        # mu input
        mu_box = QGroupBox("Parameters")
        mu_layout = QVBoxLayout()
        self.mu_input = QLineEdit("0.5")
        mu_layout.addWidget(QLabel("mu:"))
        mu_layout.addWidget(self.mu_input)
        mu_box.setLayout(mu_layout)
        left_layout.addWidget(mu_box)

        # initial condition
        ic_box = QGroupBox("Initial condition [x0, xdot0]")
        ic_layout = QVBoxLayout()
        self.ic_input = QLineEdit("0.3, 0.0")
        ic_layout.addWidget(QLabel("Enter as: x0, xdot0"))
        ic_layout.addWidget(self.ic_input)
        ic_box.setLayout(ic_layout)
        left_layout.addWidget(ic_box)

        # slider for t_end
        slider_box = QGroupBox("Simulation end time")
        slider_layout = QVBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(5)
        self.slider.setMaximum(200)
        self.slider.setValue(40)
        self.slider.valueChanged.connect(self.update_slider_label)
        self.slider_label = QLabel("End time: 40")
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.slider_label)
        slider_box.setLayout(slider_layout)
        left_layout.addWidget(slider_box)

        # run button
        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.clicked.connect(self.run_simulation)
        left_layout.addWidget(self.run_btn)

        # output text area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        left_layout.addWidget(self.output_text)

        # add left_layout to main
        main_layout.addLayout(left_layout, 1)   # stretch: left 1

        # Right panel: canvas with 2x2 plots
        self.fig = plt.figure(figsize=(10, 8))

        # create axes up front and store
        # We use plt.figure + add_subplot so we can control axes types (all 2D here)
        self.ax00 = self.fig.add_subplot(2, 2, 1)  # nonlinear time series
        self.ax01 = self.fig.add_subplot(2, 2, 2)  # phase plane (nonlinear)
        self.ax10 = self.fig.add_subplot(2, 2, 3)  # linear time series
        self.ax11 = self.fig.add_subplot(2, 2, 4)  # phase plane (linear)

        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, 4)      # stretch: right 4 (bigger)

    def update_slider_label(self):
        self.slider_label.setText(f"End time: {self.slider.value()}")

    def run_simulation(self):
        try:
            mu = float(self.mu_input.text())
            ic_text = self.ic_input.text()
            x0_str, x1_str = map(str.strip, ic_text.split(","))
            x0 = float(x0_str); x1 = float(x1_str)
            x0_vec = [x0, x1]
            t_end = float(self.slider.value())

            # run simulate
            res = simulate_vdp(mu, x0_vec, t_end)

            # clear axes
            for ax in [self.ax00, self.ax01, self.ax10, self.ax11]:
                ax.clear()

            # Unpack
            jac = res["jacobian"]
            vals = res["eigvals"]
            vecs = res["eigvecs"]
            vecs_norm = res["eigvecs_norm"]
            sol_nl = res["sol_nonlinear"]
            sol_lin = res["sol_linear"]

            # 1) nonlinear time series
            self.ax00.plot(sol_nl.t, sol_nl.y[0], label="x (position)")
            self.ax00.plot(sol_nl.t, sol_nl.y[1], '--', label="x_dot (velocity)")
            self.ax00.set_xlabel("t"); self.ax00.set_ylabel("values")
            self.ax00.set_title("Nonlinear time series")
            self.ax00.grid(True)
            self.ax00.legend()

            # 2) nonlinear phase plane w/ vector field
            # create grid spanning a bit beyond the trajectory extents
            x_min, x_max = np.min(sol_nl.y[0]), np.max(sol_nl.y[0])
            y_min, y_max = np.min(sol_nl.y[1]), np.max(sol_nl.y[1])
            pad_x = max(0.5, 0.2*(x_max - x_min if x_max> x_min else 1.0))
            pad_y = max(0.5, 0.2*(y_max - y_min if y_max> y_min else 1.0))
            xv = np.linspace(x_min - pad_x, x_max + pad_x, 25)
            yv = np.linspace(y_min - pad_y, y_max + pad_y, 25)
            Xg, Yg = np.meshgrid(xv, yv)
            Ug = Yg
            Vg = mu * (1 - Xg**2) * Yg - Xg
            speed = np.sqrt(Ug**2 + Vg**2)
            speed[speed == 0] = 1.0
            self.ax01.quiver(Xg, Yg, Ug/speed, Vg/speed, angles='xy', scale=30, alpha=0.6)
            self.ax01.plot(sol_nl.y[0], sol_nl.y[1], color='C1', label=f'IC={x0_vec}')
            self.ax01.plot(0,0,'ko')  # equilibrium
            self.ax01.annotate('(0,0)', xy=(0,0), xytext=(0, 0.2))
            self.ax01.set_xlabel('x'); self.ax01.set_ylabel('x_dot')
            self.ax01.set_title('Phase plane (Nonlinear)')
            self.ax01.grid(True)
            self.ax01.legend()

            # 3) linear time series
            self.ax10.plot(sol_lin.t, sol_lin.y[0], label='x (linear)')
            self.ax10.plot(sol_lin.t, sol_lin.y[1], '--', label='x_dot (linear)')
            self.ax10.set_xlabel('t'); self.ax10.set_ylabel('values')
            self.ax10.set_title('Linearized time series (Jacobian at origin)')
            self.ax10.grid(True)
            self.ax10.legend()

            # 4) linear phase plane (trajectories from linear system)
            self.ax11.plot(sol_lin.y[0], sol_lin.y[1], color='C2', label=f'IC={x0_vec}')
            self.ax11.plot(0,0,'ko')
            self.ax11.annotate('(0,0)', xy=(0,0), xytext=(0, 0.2))
            self.ax11.set_xlabel('x'); self.ax11.set_ylabel('x_dot')
            self.ax11.set_title('Phase plane (Linearized)')
            self.ax11.grid(True)
            self.ax11.legend()

            # draw canvas
            self.canvas.draw()

            # Update left panel outputs (nicely formatted)
            out = []
            out.append(f"Jacobian at (0,0):\n{np.array2string(jac, formatter={'float_kind':lambda x: f'{x: .4f}'})}\n")
            out.append("Eigenvalues:\n" + "\n".join([f"  {i+1}: {val:.6g}" for i,val in enumerate(vals)]) + "\n")
            out.append("Eigenvectors (columns):\n" + np.array2string(vecs, precision=4, suppress_small=True) + "\n")
            out.append("Normalized eigenvectors (columns):\n" + np.array2string(vecs_norm, precision=4, suppress_small=True))
            self.output_text.setPlainText("\n".join(out))

        except Exception as e:
            self.output_text.setPlainText(f"Error: {e}")

# ------------------------------
# Run app
# ------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VDPApp()
    win.show()
    sys.exit(app.exec_())
