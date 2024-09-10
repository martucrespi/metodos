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

def create_meshgrid(n_points):
    x = np.linspace(-1, 1, n_points)
    y = np.linspace(-1, 1, n_points)
    return np.meshgrid(x, y)

def plot_interpolation_3d(n_points):
    x, y = create_meshgrid(n_points)
    z = f2(x, y)

    points = np.vstack([x.ravel(), y.ravel()]).T
    values = z.ravel()

    xi, yi = create_meshgrid(500)
    zi = griddata(points, values, (xi, yi), method='linear')

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none')

    ax.set_title(f'Interpolación Lineal en 3D con {n_points}x{n_points} Puntos')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f2(x1, x2)')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()

for n_points in [5, 10, 50, 100]:
    plot_interpolation_3d(n_points)

