import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy.linalg as la
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSlider, QTextEdit, QGroupBox
)
from PyQt5.QtCore import Qt
import ast

# ------------------------------
# Lotka-Volterra Model Functions
# ------------------------------
def lv_rhs(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha*x - beta*x*y
    dydt = delta*x*y - gamma*y
    return [dxdt, dydt]

def simulate_lv(alpha, beta, delta, gamma, ic_list, t_end):
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, int(t_end*20))
    
    solutions = [solve_ivp(lambda t,z: lv_rhs(t,z,alpha,beta,delta,gamma),
                           t_span, ic, t_eval=t_eval,
                           rtol=1e-8, atol=1e-10)
                 for ic in ic_list]
    
    # Grid for vector field
    x_vals = np.linspace(-0.5, max([sol.y[0].max() for sol in solutions])+2, 20)
    y_vals = np.linspace(-0.5, max([sol.y[1].max() for sol in solutions])+2, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    U = alpha*X - beta*X*Y
    V = delta*X*Y - gamma*Y
    speed = np.sqrt(U**2 + V**2)
    speed[speed==0] = 1.0

    # Fixed points
    x_fp = gamma/delta
    y_fp = alpha/beta
    fixed_points = [[0.0, 0.0], [x_fp, y_fp]]

    # Jacobians
    def jacobian(x, y):
        return np.array([[alpha - beta*y, -beta*x],
                         [delta*y, delta*x - gamma]])
    jac_fp1 = jacobian(0,0)
    jac_fp2 = jacobian(x_fp, y_fp)
    
    # Eigenvalues 3x repeated
    eigvals_fp1_runs = [la.eigvals(jac_fp1) for _ in range(3)]
    eigvals_fp2_runs = [la.eigvals(jac_fp2) for _ in range(3)]
    
    return solutions, (X,Y,U,V,speed), fixed_points, (jac_fp1, jac_fp2), (eigvals_fp1_runs, eigvals_fp2_runs), t_eval

# ------------------------------
# PyQt5 GUI
# ------------------------------
class LVApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lotka-Volterra Simulator")
        self.setGeometry(50,50,1600,900)

        layout = QHBoxLayout(self)

        # Left panel: Inputs + Outputs
        control_layout = QVBoxLayout()

        # Parameters
        param_box = QGroupBox("Model Parameters")
        param_layout = QVBoxLayout()
        self.alpha_input = QLineEdit("1.1")
        self.beta_input = QLineEdit("0.4")
        self.delta_input = QLineEdit("0.1")
        self.gamma_input = QLineEdit("0.4")
        for label, widget in zip(["alpha:", "beta:", "delta:", "gamma:"],
                                 [self.alpha_input, self.beta_input, self.delta_input, self.gamma_input]):
            param_layout.addWidget(QLabel(label))
            param_layout.addWidget(widget)
        param_box.setLayout(param_layout)
        control_layout.addWidget(param_box)

        # Initial Conditions
        ic_box = QGroupBox("Initial Conditions (x,y)")
        ic_layout = QVBoxLayout()
        self.ic_input = QLineEdit("[[10,5],[5,2]]")
        ic_layout.addWidget(QLabel("ICs as list:"))
        ic_layout.addWidget(self.ic_input)
        ic_box.setLayout(ic_layout)
        control_layout.addWidget(ic_box)

        # Slider for t_end
        slider_box = QGroupBox("Simulation Time")
        slider_layout = QVBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(10)
        self.slider.setMaximum(200)
        self.slider.setValue(60)
        self.slider.valueChanged.connect(self.update_slider_label)
        self.slider_label = QLabel("End time: 60")
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
        self.fig, self.axs = plt.subplots(1,2, figsize=(13,6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas,4)

    def update_slider_label(self):
        self.slider_label.setText(f"End time: {self.slider.value()}")

    def run_simulation(self):
        try:
            alpha = float(self.alpha_input.text())
            beta  = float(self.beta_input.text())
            delta = float(self.delta_input.text())
            gamma = float(self.gamma_input.text())
            t_end = self.slider.value()
            ic_list = np.array(ast.literal_eval(self.ic_input.text()))

            solutions, (X,Y,U,V,speed), fixed_points, (jac_fp1,jac_fp2), (eig_fp1,eig_fp2), t_eval = simulate_lv(alpha,beta,delta,gamma,ic_list,t_end)

            # Clear previous plots
            for ax in self.axs:
                ax.clear()

            # --------------------
            # Time evolution
            # --------------------
            ax = self.axs[0]
            for sol, ic in zip(solutions, ic_list):
                ax.plot(sol.t, sol.y[0], label=f'x IC={ic}')
                ax.plot(sol.t, sol.y[1], '--', label=f'y IC={ic}')
            ax.set_xlabel('t'); ax.set_ylabel('Population')
            ax.set_title("Lotka-Volterra: Prey solid, Predator dashed")
            ax.grid(True)
            ax.legend(fontsize='small')

            # --------------------
            # Phase plane
            # --------------------
            ax = self.axs[1]
            ax.quiver(X,Y,U/speed,V/speed,angles='xy',scale=50,width=0.002,alpha=0.7)
            for sol, ic in zip(solutions, ic_list):
                ax.plot(sol.y[0], sol.y[1], label=f'IC={ic}')

            # Nullclines
            x_null = gamma/delta
            y_null = alpha/beta
            ax.axvline(x_null, linestyle='--', color='red')
            ax.axhline(y_null, linestyle='--', color='blue')

            # Fixed points
            for x_fp, y_fp in fixed_points:
                ax.plot(x_fp, y_fp, 'ko', markersize=6)
                ax.annotate(f'({x_fp:.2f},{y_fp:.2f})', xy=(x_fp, y_fp), xytext=(x_fp+0.5,y_fp+0.5),
                            arrowprops=dict(facecolor='black', arrowstyle='->'))

            ax.set_xlabel("x (prey)"); ax.set_ylabel("y (predator)")
            ax.set_xlim(-1,max([sol.y[0].max() for sol in solutions])+0.1*max([sol.y[0].max() for sol in solutions]))
            ax.set_ylim(-1,max([sol.y[1].max() for sol in solutions])+0.1*max([sol.y[1].max() for sol in solutions]))
            ax.set_title("Phase plane: trajectories, vector field, nullclines")
            ax.grid(True)
            ax.legend(fontsize='small')

            self.canvas.draw()

            # --------------------
            # Output panel
            # --------------------
            output = f"Fixed points:\n (0,0) and ({x_null:.4f},{y_null:.4f})\n\n"
            output += f"Jacobian at (0,0):\n{np.round(jac_fp1,4)}\n"
            output += f"Eigenvalues (3 runs):\n{np.round(eig_fp1,4)}\n\n"
            output += f"Jacobian at ({x_null:.4f},{y_null:.4f}):\n{np.round(jac_fp2,4)}\n"
            output += f"Eigenvalues (3 runs):\n{np.round(eig_fp2,4)}"
            self.output_text.setText(output)

        except Exception as ex:
            self.output_text.setText(f"Error: {ex}")

# ------------------------------
# Run application
# ------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LVApp()
    window.show()
    sys.exit(app.exec_())
