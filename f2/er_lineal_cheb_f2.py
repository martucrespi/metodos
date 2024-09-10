import numpy as np
from scipy.interpolate import griddata

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

def calculate_absolute_error(exact, interp):
    interp_clean = np.nan_to_num(interp, nan=0.0)
    return np.abs(exact - interp_clean)

def get_error_stats(error):
    max_error = np.max(error)
    median_error = np.median(error)
    return max_error, median_error

def calculate_error_stats_chebyshev_linear(n_points_1, n_points_2):
    xi, yi = np.meshgrid(np.linspace(-1, 1, 500), np.linspace(-1, 1, 500))
    z_exact = f2(xi, yi)

    x1, y1 = create_chebyshev_meshgrid(n_points_1)
    z1 = f2(x1, y1)
    points1 = np.vstack([x1.ravel(), y1.ravel()]).T
    z_interp_1 = griddata(points1, z1.ravel(), (xi, yi), method='linear')

    x2, y2 = create_chebyshev_meshgrid(n_points_2)
    z2 = f2(x2, y2)
    points2 = np.vstack([x2.ravel(), y2.ravel()]).T
    z_interp_2 = griddata(points2, z2.ravel(), (xi, yi), method='linear')

    error_1 = calculate_absolute_error(z_exact, z_interp_1)
    error_2 = calculate_absolute_error(z_exact, z_interp_2)

    max_error_1, median_error_1 = get_error_stats(error_1)
    max_error_2, median_error_2 = get_error_stats(error_2)

    print(f"Con {n_points_1} puntos (Chebyshev, Interpolación Lineal):")
    print(f"  Error máximo: {max_error_1:.6f}")
    print(f"  Mediana del error: {median_error_1:.6f}")

    print(f"\nCon {n_points_2} puntos (Chebyshev, Interpolación Lineal):")
    print(f"  Error máximo: {max_error_2:.6f}")
    print(f"  Mediana del error: {median_error_2:.6f}")

calculate_error_stats_chebyshev_linear(5, 50)
