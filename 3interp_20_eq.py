import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, lagrange, CubicSpline

def f1(x):
    return -0.4 * np.tanh(50 * x) + 0.6

x_range = np.linspace(-1, 1, 1000)
y_true = f1(x_range)

num_points_linear_spline = 20

x_points_linear_spline = np.linspace(-1, 1, num_points_linear_spline)
y_points_linear_spline = f1(x_points_linear_spline)

linear_interp = interp1d(x_points_linear_spline, y_points_linear_spline, kind='linear')
y_linear_interp = linear_interp(x_range)

spline = CubicSpline(x_points_linear_spline, y_points_linear_spline)
y_spline_interp = spline(x_range)

num_points_lagrange = 5

x_points_lagrange = np.linspace(-1, 1, num_points_lagrange)
y_points_lagrange = f1(x_points_lagrange)

lagrange_poly = lagrange(x_points_lagrange, y_points_lagrange)
y_lagrange_interp = lagrange_poly(x_range)

plt.figure(figsize=(10, 6))

plt.plot(x_range, y_true, label='Función original', color='blue')

plt.plot(x_range, y_linear_interp, '--', label='Interpolación lineal con 20 puntos', color='red')
plt.plot(x_range, y_spline_interp, '-.', label='Interpolación de splines cúbicos con 20 puntos', color='orange')
plt.plot(x_range, y_lagrange_interp, ':', label='Interpolación de Lagrange con 5 puntos', color='green')

plt.title('Comparación de Interpolaciones con 5 puntos equiespaciados')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()