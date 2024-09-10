import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def f1(x):
    return -0.4 * np.tanh(50 * x) + 0.6

x_values = np.linspace(-1, 1, 1000)
y_values = f1(x_values)

points_to_compare = [20, 200]

plt.figure(figsize=(10, 6))

for n_points in points_to_compare:
    x_interp = np.linspace(-1, 1, n_points)
    y_interp = f1(x_interp)
    
    spline_interp = CubicSpline(x_interp, y_interp)
    y_spline_interp = spline_interp(x_values)
    
    relative_error = np.abs((y_values - y_spline_interp) / y_values)
    
    plt.plot(x_values, relative_error, label=f"Error relativo con {n_points} puntos")

plt.legend()
plt.title("Error relativo de interpolación de Splines Cúbicos con 20 y 200 puntos")
plt.xlabel("x")
plt.ylabel("Error relativo")
plt.grid(True)
plt.show()
