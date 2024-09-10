import numpy as np
from scipy.interpolate import griddata

# Definición de la función f2
def f2(x1, x2):
    term1 = 0.75 * np.exp(-((9 * x1 - 2)**2 / 4) - ((9 * x2 - 2)**2 / 4))
    term2 = 0.75 * np.exp(-((9 * x1 + 1)**2 / 49) - ((9 * x2 + 1)**2 / 10))
    term3 = 0.5 * np.exp(-((9 * x1 - 7)**2 / 4) - ((9 * x2 - 3)**2 / 4))
    term4 = -0.2 * np.exp(-((9 * x1 - 7)**2 / 4) - ((9 * x2 - 3)**2 / 4))
    return term1 + term2 + term3 + term4

# Crear la malla para el intervalo [-1, 1] con puntos equiespaciados
def create_meshgrid(n_points):
    x = np.linspace(-1, 1, n_points)
    y = np.linspace(-1, 1, n_points)
    return np.meshgrid(x, y)

# Calcular el error absoluto
def calculate_absolute_error(exact, interp):
    # Reemplazar NaN en los resultados interpolados
    interp_clean = np.nan_to_num(interp, nan=0.0)
    return np.abs(exact - interp_clean)

# Función para calcular el error máximo y la mediana
def get_error_stats(error):
    max_error = np.max(error)
    median_error = np.median(error)
    return max_error, median_error

# Calcular e imprimir el error relativo para dos mallas de puntos equiespaciados
def calculate_error_stats_linear(n_points_1, n_points_2):
    # Malla fina de referencia (equiespaciada)
    xi, yi = create_meshgrid(500)
    z_exact = f2(xi, yi)

    # Interpolación con n_points_1 usando puntos equiespaciados
    x1, y1 = create_meshgrid(n_points_1)
    z1 = f2(x1, y1)
    points1 = np.vstack([x1.ravel(), y1.ravel()]).T
    z_interp_1 = griddata(points1, z1.ravel(), (xi, yi), method='linear')

    # Interpolación con n_points_2 usando puntos equiespaciados
    x2, y2 = create_meshgrid(n_points_2)
    z2 = f2(x2, y2)
    points2 = np.vstack([x2.ravel(), y2.ravel()]).T
    z_interp_2 = griddata(points2, z2.ravel(), (xi, yi), method='linear')

    # Calcular el error relativo
    error_1 = calculate_absolute_error(z_exact, z_interp_1)
    error_2 = calculate_absolute_error(z_exact, z_interp_2)

    # Calcular el error máximo y la mediana para ambos casos
    max_error_1, median_error_1 = get_error_stats(error_1)
    max_error_2, median_error_2 = get_error_stats(error_2)

    print(f"Con {n_points_1} puntos (Equiespaciados, Interpolación Lineal):")
    print(f"  Error máximo: {max_error_1:.6f}")
    print(f"  Mediana del error: {median_error_1:.6f}")

    print(f"\nCon {n_points_2} puntos (Equiespaciados, Interpolación Lineal):")
    print(f"  Error máximo: {max_error_2:.6f}")
    print(f"  Mediana del error: {median_error_2:.6f}")

# Calcular e imprimir los errores para 5 y 50 puntos equiespaciados
calculate_error_stats_linear(5, 50)
