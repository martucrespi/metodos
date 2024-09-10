import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

def f2(x1, x2):
    term1 = 0.75 * np.exp(-((9 * x1 - 2) ** 2 / 4) - ((9 * x2 - 2) ** 2 / 4))
    term2 = 0.75 * np.exp(-((9 * x1 + 1) ** 2 / 49) - ((9 * x2 + 1) ** 2 / 10))
    term3 = 0.5 * np.exp(-((9 * x1 - 7) ** 2 / 4) - ((9 * x2 - 3) ** 2 / 4))
    term4 = -0.2 * np.exp(-((9 * x1 - 7) ** 2 / 4) - ((9 * x2 - 3) ** 2 / 4))
    return term1 + term2 + term3 + term4

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)

def generate_chebyshev_grid_data(n_points):
    chebyshev_points = np.cos(np.pi * (2 * np.arange(n_points) + 1) / (2 * n_points))
    x = np.sort(chebyshev_points)
    y = np.sort(chebyshev_points)
    X, Y = np.meshgrid(x, y)
    Z = f2(X, Y)
    return X, Y, Z

def interpolate_and_plot_3d_chebyshev(n_points):
    Xg, Yg, Zg = generate_chebyshev_grid_data(n_points)
    points = np.vstack((Xg.ravel(), Yg.ravel())).T
    values = Zg.ravel()
    Z_interp = griddata(points, values, (X, Y), method='cubic')
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z_interp, cmap='viridis', edgecolor='none')
    ax.set_title(f'Interpolación con {n_points}x{n_points} puntos (Chebyshev)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f2(x1, x2)')
    plt.show()

interpolate_and_plot_3d_chebyshev(5)
interpolate_and_plot_3d_chebyshev(10)
interpolate_and_plot_3d_chebyshev(50)
interpolate_and_plot_3d_chebyshev(100)
