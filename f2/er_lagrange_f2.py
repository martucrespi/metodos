import numpy as np

# Definición de la función f2 (ya dada)
def f2(x1, x2):
    term1 = 0.75 * np.exp(-((9 * x1 - 2)**2 / 4) - ((9 * x2 - 2)**2 / 4))
    term2 = 0.75 * np.exp(-((9 * x1 + 1)**2 / 49) - ((9 * x2 + 1)**2 / 10))
    term3 = 0.5 * np.exp(-((9 * x1 - 7)**2 / 4) - ((9 * x2 - 3)**2 / 4))
    term4 = -0.2 * np.exp(-((9 * x1 - 7)**2 / 4) - ((9 * x2 - 3)**2 / 4))
    return term1 + term2 + term3 + term4

# Crear la malla para el intervalo [-1, 1]
def create_meshgrid(n_points):
    x = np.linspace(-1, 1, n_points)
    y = np.linspace(-1, 1, n_points)
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

# Calcular el error absoluto
def calculate_absolute_error(exact, interp):
    return np.abs(exact - interp)

# Función para calcular el error máximo y la mediana
def get_error_stats(error):
    max_error = np.max(error)
    median_error = np.median(error)
    return max_error, median_error

# Calcular e imprimir el error relativo para dos mallas de puntos
def calculate_error_stats(n_points_1, n_points_2):
    # Malla fina de referencia
    xi, yi = create_meshgrid(500)
    z_exact = f2(xi, yi)

    # Interpolación con n_points_1
    x1, y1 = create_meshgrid(n_points_1)
    z1 = f2(x1, y1)
    z_interp_1 = lagrange_interpolation_2d(xi, yi, x1[0], y1[:,0], z1)

    # Interpolación con n_points_2
    x2, y2 = create_meshgrid(n_points_2)
    z2 = f2(x2, y2)
    z_interp_2 = lagrange_interpolation_2d(xi, yi, x2[0], y2[:,0], z2)

    # Calcular el error relativo
    error_1 = calculate_absolute_error(z_exact, z_interp_1)
    error_2 = calculate_absolute_error(z_exact, z_interp_2)

    # Calcular el error máximo y la mediana para ambos casos
    max_error_1, median_error_1 = get_error_stats(error_1)
    max_error_2, median_error_2 = get_error_stats(error_2)

    print(f"Con {n_points_1} puntos (Equiespaciados, Lagrange):")
    print(f"  Error máximo: {max_error_1:.6f}")
    print(f"  Mediana del error: {median_error_1:.6f}")

    print(f"\nCon {n_points_2} puntos (Equiespaciados, Lagrange):")
    print(f"  Error máximo: {max_error_2:.6f}")
    print(f"  Mediana del error: {median_error_2:.6f}")

# Calcular e imprimir los errores para 5 puntos y 20 puntos
calculate_error_stats(5, 20)
