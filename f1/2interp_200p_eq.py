import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline, lagrange

def f1(x):
    return -0.4 * np.tanh(50 * x) + 0.6

x_range = np.linspace(-1, 1, 1000)
y_true = f1(x_range)

num_points = 200

x_points = np.linspace(-1, 1, num_points)
y_points = f1(x_points)

linear_interp = interp1d(x_points, y_points, kind='linear')
y_linear_interp = linear_interp(x_range)

spline = CubicSpline(x_points, y_points)
y_spline_interp = spline(x_range)

num_points_lagrange = 20
x_points_lagrange = np.linspace(-1, 1, num_points_lagrange)
y_points_lagrange = f1(x_points_lagrange)
polynomial_lagrange = lagrange(x_points_lagrange, y_points_lagrange)
y_lagrange_interp = polynomial_lagrange(x_range)

plt.figure(figsize=(10, 6))

plt.plot(x_range, y_true, label='Función original', color='blue')
plt.plot(x_range, y_linear_interp, '--', label='Interpolación lineal con 200 puntos', color='red')
plt.plot(x_range, y_spline_interp, '-.', label='Interpolación de splines cúbicos con 200 puntos', color='orange')
plt.plot(x_range, y_lagrange_interp, ':', label='Interpolación de Lagrange con 20 puntos', color='green')

plt.title('Comparación de Interpolaciones con 200 y 20 puntos equiespaciados')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.ylim(0, 1.1)
plt.show()
