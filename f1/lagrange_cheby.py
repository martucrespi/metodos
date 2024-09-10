import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

def f1(x):
    return -0.4 * np.tanh(50 * x) + 0.6

def chebyshev_nodes(n):
    return np.cos((2 * np.arange(1, n + 1) - 1) / (2 * n) * np.pi)

x_values = np.linspace(-1, 1, 1000)
y_values = f1(x_values)

points_to_compare = [5, 20]

plt.figure(figsize=(10, 6))

for n_points in points_to_compare:
    x_interp = chebyshev_nodes(n_points)
    y_interp = f1(x_interp)
    
    lagrange_interp = lagrange(x_interp, y_interp)
    y_lagrange_interp = lagrange_interp(x_values)
    
    plt.plot(x_values, y_lagrange_interp, '--', label=f"Interpolación de Lagrange con {n_points} puntos (Chebyshev)")
    
    plt.scatter(x_interp, y_interp, s=15, label=f'Puntos de interpolación ({n_points} puntos)')

plt.plot(x_values, y_values, label="Función original", color="blue")

plt.ylim(0, 1.5)

plt.legend()
plt.title(f"Interpolación de Lagrange con 5 y 20 puntos de Chebyshev")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()
