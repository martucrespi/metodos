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

def generate_grid_data(n_points):
    x = np.linspace(-1, 1, n_points)
    y = np.linspace(-1, 1, n_points)
    X, Y = np.meshgrid(x, y)
    Z = f2(X, Y)
    return X, Y, Z

def calculate_absolute_error(exact, interp):
    interp_clean = np.nan_to_num(interp, nan=0.0)
    return np.abs(exact - interp_clean)

def get_error_stats(error):
    max_error = np.max(error)
    median_error = np.median(error)
    return max_error, median_error

def interpolate_and_calculate_errors(n_points, fine_grid_size=500):
    Xg, Yg, Zg = generate_grid_data(n_points)
    points = np.vstack((Xg.ravel(), Yg.ravel())).T
    values = Zg.ravel()

    x_fine = np.linspace(-1, 1, fine_grid_size)
    y_fine = np.linspace(-1, 1, fine_grid_size)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    Z_exact = f2(X_fine, Y_fine)

    Z_interp = griddata(points, values, (X_fine, Y_fine), method='cubic')
    error = calculate_absolute_error(Z_exact, Z_interp)
    max_error, median_error = get_error_stats(error)

    print(f"Con {n_points} puntos (Equiespaciados, Interpolación Cúbica):")
    print(f"  Error máximo: {max_error:.6f}")
    print(f"  Mediana del error: {median_error:.6f}")

interpolate_and_calculate_errors(5)
interpolate_and_calculate_errors(100)
