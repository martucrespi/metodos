import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necesario para gráficos 3D

# Definición de la función f2
def f2(x1, x2):
    term1 = 0.75 * np.exp(-((9 * x1 - 2)**2 / 4) - ((9 * x2 - 2)**2 / 4))
    term2 = 0.75 * np.exp(-((9 * x1 + 1)**2 / 49) - ((9 * x2 + 1)**2 / 10))
    term3 = 0.5 * np.exp(-((9 * x1 - 7)**2 / 4) - ((9 * x2 - 3)**2 / 4))
    term4 = -0.2 * np.exp(-((9 * x1 - 7)**2 / 4) - ((9 * x2 - 3)**2 / 4))
    return term1 + term2 + term3 + term4

# Generar puntos de Chebyshev
def chebyshev_points(n_points):
    return np.cos((2 * np.arange(n_points) + 1) * np.pi / (2 * n_points))

# Crear la malla con puntos de Chebyshev en [-1, 1]
def create_chebyshev_meshgrid(n_points):
    x = chebyshev_points(n_points)
    y = chebyshev_points(n_points)
    return np.meshgrid(x, y)

# Función de Lagrange en 1D
def lagrange_basis_1d(x, x_values, k):
    L_k = np.ones_like(x)
    for j in range(len(x_values)):
        if j != k:
            L_k *= (x - x_values[j]) / (x_values[k] - x_values[j])
    return L_k

# Interpolación de Lagrange en 2D
def lagrange_interpolation_2d(x, y, x_values, y_values, z_values):
    z_interp = np.zeros_like(x)
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            Lx = lagrange_basis_1d(x, x_values, i)
            Ly = lagrange_basis_1d(y, y_values, j)
            z_interp += z_values[i, j] * Lx * Ly
    return z_interp

# Graficación de la interpolación de Lagrange en 3D con puntos de Chebyshev
def plot_lagrange_interpolation_chebyshev_3d(n_points):
    x, y = create_chebyshev_meshgrid(n_points)
    z = f2(x, y)

    # Crear malla de interpolación fina (equiespaciada)
    xi, yi = create_chebyshev_meshgrid(500)

    # Interpolación de Lagrange
    zi = lagrange_interpolation_2d(xi, yi, x[0], y[:,0], z)

    # Crear la figura en 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar la superficie
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none')

    # Opciones de la gráfica
    ax.set_title(f'Interpolación de Lagrange en 3D con Puntos de Chebyshev ({n_points}x{n_points})')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f2(x1, x2)')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()

# Graficar la interpolación en 3D para diferentes tamaños de malla
for n_points in [5, 10, 20]:
    plot_lagrange_interpolation_chebyshev_3d(n_points)
