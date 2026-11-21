import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eig, expm
import jax.numpy as jnp
from jax import jacfwd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSlider, QTextEdit, QGroupBox
)
from PyQt5.QtCore import Qt

# ------------------------------
# Simulation Function
# ------------------------------
def simulate_damped_oscillator(zeta, x0, t_end):
    def damped_oscillator(t, X):
        x1, x2 = X
        dx1dt = x2
        dx2dt = -x1 - 2*zeta*x2
        return jnp.array([dx1dt, dx2dt])

    t_span = (0, t_end)
    t_eval = np.linspace(*t_span, 2000)
    sol_numerical = solve_ivp(damped_oscillator, t_span, x0, t_eval=t_eval)

    j_fun = jacfwd(lambda X: damped_oscillator(0, X))
    x_eq = jnp.array([0.0, 0.0])
    jacobian = j_fun(x_eq)
    vals, vecs = eig(jacobian)
    vecs_real = np.real(vecs)
    vecs_real = vecs_real / np.linalg.norm(vecs_real)

    sol_analytic = np.column_stack([expm(jacobian * tt) @ x0 for tt in t_eval])

    return sol_numerical, sol_analytic, jacobian, vals, vecs_real, t_eval

# ------------------------------
# PyQt5 App Class
# ------------------------------
class OscillatorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Damped Harmonic Oscillator Simulator")
        self.setGeometry(100, 100, 1400, 800)
        
        layout = QHBoxLayout(self)
        
        # Left: Input + Button + Slider + Output text
        control_layout = QVBoxLayout()
        
        # Input group
        input_box = QGroupBox("Simulation Parameters")
        input_layout = QVBoxLayout()

        self.zeta_input = QLineEdit("0.1")
        self.x1_input = QLineEdit("1.0")
        self.x2_input = QLineEdit("1.0")

        input_layout.addWidget(QLabel("ζ (Damping Ratio):"))
        input_layout.addWidget(self.zeta_input)
        input_layout.addWidget(QLabel("Initial x1 (Position):"))
        input_layout.addWidget(self.x1_input)
        input_layout.addWidget(QLabel("Initial x2 (Velocity):"))
        input_layout.addWidget(self.x2_input)

        input_box.setLayout(input_layout)
        control_layout.addWidget(input_box)
        
        # Slider
        slider_box = QGroupBox("Simulation Time Span")
        slider_layout = QVBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(5)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.update_slider_label)
        self.slider_label = QLabel("Simulation End Time: 50s")
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.slider_label)
        slider_box.setLayout(slider_layout)
        control_layout.addWidget(slider_box)
        
        # Button
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        control_layout.addWidget(self.run_button)
        
        # Output text area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        control_layout.addWidget(self.output_text)
        
        layout.addLayout(control_layout, 1)
        
        # Right: Plot Canvas
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 2)

    def update_slider_label(self):
        self.slider_label.setText(f"Simulation End Time: {self.slider.value()}s")

    def run_simulation(self):
        try:
            zeta = float(self.zeta_input.text())
            x1_0 = float(self.x1_input.text())
            x2_0 = float(self.x2_input.text())
            t_end = self.slider.value()
            
            x0 = [x1_0, x2_0]
            sol_numerical, sol_analytic, jacobian, vals, vecs_real, t_eval = simulate_damped_oscillator(zeta, x0, t_end)
            
            # Clear previous plots
            for ax_row in self.axs:
                for ax in ax_row:
                    ax.clear()
            
            # Time series
            self.axs[0][0].plot(sol_numerical.t, sol_numerical.y[0], label="x1 (position)")
            self.axs[0][0].plot(sol_numerical.t, sol_numerical.y[1], label="x2 (velocity)")
            self.axs[0][0].set_title("Numerical Solution")
            self.axs[0][0].legend()
            self.axs[0][0].grid()

            self.axs[0][1].plot(t_eval, sol_analytic[0, :], '--', label='x1 (position)')
            self.axs[0][1].plot(t_eval, sol_analytic[1, :], '--', label='x2 (velocity)')
            self.axs[0][1].set_title("Analytic Solution")
            self.axs[0][1].legend()
            self.axs[0][1].grid()

            # Phase plane
            x_vals = np.linspace(-1.5, 1.5, 20)
            y_vals = np.linspace(-1.5, 1.5, 20)
            Xg, Yg = np.meshgrid(x_vals, y_vals)
            Ug = Yg
            Vg = -Xg - 2*zeta*Yg
            speed = np.sqrt(Ug**2 + Vg**2); speed[speed == 0] = 1.0
            self.axs[1][0].quiver(Xg, Yg, Ug/speed, Vg/speed, angles='xy', scale=25, alpha=0.6)
            self.axs[1][0].plot(sol_numerical.y[0], sol_numerical.y[1], color='red')
            self.axs[1][0].set_title("Phase Plane (Numerical)")
            self.axs[1][0].grid()

            self.axs[1][1].quiver(Xg, Yg, Ug/speed, Vg/speed, angles='xy', scale=25, alpha=0.6)
            self.axs[1][1].plot(sol_analytic[0, :], sol_analytic[1, :], color='red')
            self.axs[1][1].set_title("Phase Plane (Analytic)")
            self.axs[1][1].grid()

            self.fig.suptitle(f"Damped Harmonic Oscillator (ζ = {zeta})")
            self.canvas.draw()

            # Display numerical details
            text = (f"Jacobian:\n{jacobian}\n\n"
                    f"Eigenvalues:\n{vals}\n\n"
                    f"Eigenvectors (normalized):\n{vecs_real}")
            self.output_text.setPlainText(text)
        except Exception as e:
            self.output_text.setPlainText(f"Error: {str(e)}")

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OscillatorApp()
    window.show()
    sys.exit(app.exec_())
