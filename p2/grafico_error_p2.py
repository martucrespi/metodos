import numpy as np
from scipy.optimize import root
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pandas as pd

def funciones(xyz, sensores, distancias):
    x, y, z = xyz
    return [
        (x - x1)**2 + (y - y1)**2 + (z - z1)**2 - d**2 
        for (x1, y1, z1), d in zip(sensores, distancias)
    ]

def jacobiano(xyz, sensores):
    x, y, z = xyz
    J = np.zeros((len(sensores), 3))
    for i, (x1, y1, z1) in enumerate(sensores):
        J[i] = [
            2 * (x - x1), 
            2 * (y - y1), 
            2 * (z - z1)
        ]
    return J

def newton_raphson_3d(sensores, distancias, estimacion_inicial, tol=1e-6, max_iter=100):
    xyz = np.array(estimacion_inicial, dtype=np.float64)
    for _ in range(max_iter):
        F = funciones(xyz, sensores, distancias)
        J = jacobiano(xyz, sensores)
        
        delta = np.linalg.solve(J, -np.array(F))
        xyz += delta
        
        if np.linalg.norm(delta) < tol:
            break
    return xyz

measurements_data = np.loadtxt('/Users/sofiab/Métodos Numéricos/measurements.txt', delimiter=',', skiprows=1)
sensor_positions_data = np.loadtxt('/Users/sofiab/Métodos Numéricos/sensor_positions.txt', delimiter=',', skiprows=1)
trajectory_data = np.loadtxt('/Users/sofiab/Métodos Numéricos/trajectory.txt', delimiter=',', skiprows=1)

trajectory = pd.DataFrame(trajectory_data, columns=['t(s)', 'x(m)', 'y(m)', 'z(m)'])
measurements = pd.DataFrame(measurements_data, columns=['t(s)', 'd1(m)', 'd2(m)', 'd3(m)'])
sensor_positions = pd.DataFrame(sensor_positions_data, columns=['i', 'x_i(m)', 'y_i(m)', 'z_i(m)'])

sensores = sensor_positions[['x_i(m)', 'y_i(m)', 'z_i(m)']].values
estimacion_inicial = np.array([0, 0, 0], dtype=np.float64)

trayectoria_calculada = []

for index, row in measurements.iterrows():
    distancias = row[['d1(m)', 'd2(m)', 'd3(m)']].values
    posicion_calculada = newton_raphson_3d(sensores, distancias, estimacion_inicial)
    trayectoria_calculada.append((row['t(s)'], *posicion_calculada))

trayectoria_calculada = np.array(trayectoria_calculada)
tiempos = trayectoria_calculada[:, 0]
posiciones_x = trayectoria_calculada[:, 1]
posiciones_y = trayectoria_calculada[:, 2]
posiciones_z = trayectoria_calculada[:, 3]

cs_x = CubicSpline(tiempos, posiciones_x)
cs_y = CubicSpline(tiempos, posiciones_y)
cs_z = CubicSpline(tiempos, posiciones_z)

t_interpolado = trajectory['t(s)'].values
real_interp_x = trajectory['x(m)'].values
real_interp_y = trajectory['y(m)'].values
real_interp_z = trajectory['z(m)'].values

posicion_interpolada_x = cs_x(t_interpolado)
posicion_interpolada_y = cs_y(t_interpolado)
posicion_interpolada_z = cs_z(t_interpolado)

errores = np.sqrt((posicion_interpolada_x - real_interp_x)**2 + 
                  (posicion_interpolada_y - real_interp_y)**2 + 
                  (posicion_interpolada_z - real_interp_z)**2)

mediana_error = np.median(errores)
maximo_error = np.max(errores)

fig = plt.figure(figsize=(10, 8))
plt.plot(t_interpolado, errores, label='Error', linewidth=1.5, color='#c0392b')
plt.xlabel('Tiempo (s)', fontsize=10)
plt.ylabel('Error (m)', fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.title('Error absoluto de la interpolación en función del tiempo', fontsize=15)
plt.text(0.5, 0.8, f"Mediana del error: {mediana_error:.4f} m\nError máximo: {maximo_error:.4f} m", 
          transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'), 
          fontsize=10, ha='left')
plt.legend(fontsize=10)
plt.show()

