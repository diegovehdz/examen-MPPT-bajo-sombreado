"""
exploración de las curvas P-I bajo sombreado parcial
para nosotros ver el ground-truth con fines de análisis (el agente nunca debe usarlo!!!!!).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- 1. dataa
# cada archivo es una tabla (N, 3) con columnas 
# [corriente A, voltaje V, potencia W]
casos = [np.load(f"datasets/caso_{k}.npy") for k in range(4)]

for k, datos in enumerate(casos):
    I, V, P = datos[:, 0], datos[:, 1], datos[:, 2]
    print(f"case {k}: {len(datos)} puntos | I_sc = {I.max():.2f} A | "
          f"V_oc = {V.max():.2f} V | P_max = {P.max():.2f} W")
    
# --- 2. convertir la tabla en una función P(I)
# acá hacemos la interpolación de la tabla para poder consultar la potencia a cualquier corriente
# con un promedio de los puntos repetidos
def curva_PI(datos):
    I, P = datos[:, 0], datos[:, 2]
    I_unica, idx = np.unique(I, return_inverse=True)           # ordenar y juntar repetidos
    P_unica = np.bincount(idx, weights=P) / np.bincount(idx)   # promediar los repetidos
    
    # agregar el punto (0 A, 0 W)
    # en circuito abierto no hay potencia lol
    return np.concatenate([[0.0], I_unica]), np.concatenate([[0.0], P_unica])


def potencia(I_consulta, I_tabla, P_tabla):
    # right = 0.0 significa que si pides más corriente que I_sc entonces el panel no entrega potencia
    # gracias google por
    return np.interp(I_consulta, I_tabla, P_tabla, right=0.0)