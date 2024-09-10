import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D 

def f2(x1, x2):
    term1 = 0.75 * np.exp(-((9 * x1 - 2)**2 / 4) - ((9 * x2 - 2)**2 / 4))
    term2 = 0.75 * np.exp(-((9 * x1 + 1)**2 / 49) - ((9 * x2 + 1)**2 / 10))
    term3 = 0.5 * np.exp(-((9 * x1 - 7)**2 / 4) - ((9 * x2 - 3)**2 / 4))
    term4 = -0.2 * np.exp(-((9 * x1 - 7)**2 / 4) - ((9 * x2 - 3)**2 / 4))
    return term1 + term2 + term3 + term4

def chebyshev_points(n_points):
    return np.cos((2 * np.arange(n_points) + 1) * np.pi / (2 * n_points))

def create_chebyshev_meshgrid(n_points):
    x = chebyshev_points(n_points)
    y = chebyshev_points(n_points)
    return np.meshgrid(x, y)

def plot_interpolation_chebyshev_3d(n_points):
    x, y = create_chebyshev_meshgrid(n_points)
    z = f2(x, y)

    points = np.vstack([x.ravel(), y.ravel()]).T
    values = z.ravel()

    xi, yi = create_chebyshev_meshgrid(500)
    zi = griddata(points, values, (xi, yi), method='linear')

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none')

    ax.scatter(x, y, z, color='red', s=2, label=f'Puntos {n_points}x{n_points}')

    ax.set_title(f'Interpolación Lineal con Puntos de Chebyshev ({n_points}x{n_points})')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f2(x1, x2)')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.legend()

    plt.show()

for n_points in [5, 10, 50]:
    plot_interpolation_chebyshev_3d(n_points)
