import matplotlib.pyplot as plt
import numpy as np

metodos = [
    '5 pts Equiespaciados\nInterpolación Lineal',
    '50 pts Equiespaciados\nInterpolación Lineal',
    '5 pts Chebyshev\nInterpolación Lineal',
    '50 pts Chebyshev\nInterpolación Lineal',
    '5 pts Equiespaciados\nLagrange',
    '20 pts Equiespaciados\nLagrange',
    '5 pts Chebyshev\nLagrange',
    '20 pts Chebyshev\nLagrange',
    '5 pts Equiespaciados\nInterpolación Cúbica',
    '100 pts Equiespaciados\nInterpolación Cúbica',
    '5 pts Chebyshev\nInterpolación Cúbica',
    '100 pts Chebyshev\nInterpolación Cúbica'
]

errores_maximos = [
    0.688791, 0.012035, 0.681512, 0.203150, 
    0.557098, 3.009393, 0.486646, 0.397571, 
    0.603041, 0.000437, 0.589247, 0.203150
]

mediana_errores = [
    0.033471, 0.000241, 0.023333, 0.000389, 
    0.148262, 0.140866, 0.129861, 0.126101, 
    0.023142, 0.000004, 0.018505, 0.000006
]

x = np.arange(len(metodos))

plt.figure(figsize=(10, 6))
plt.bar(x, errores_maximos, color='skyblue')
plt.xticks(x, metodos, rotation=90)
plt.ylabel('Error Máximo')
plt.title('Errores Máximos por Método de Interpolación')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x, mediana_errores, color='lightgreen')
plt.xticks(x, metodos, rotation=90)
plt.ylabel('Mediana del Error')
plt.title('Mediana de Errores por Método de Interpolación')
plt.tight_layout()
plt.show()
