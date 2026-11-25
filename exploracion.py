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

# --- 3. graf
I_max = max(d[:, 0].max() for d in casos) # el max que vi fue como 8.90 A en los case 1 y 3
I_grid = np.linspace(0, I_max, 1000)

fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for k, (ax, datos) in enumerate(zip(axs.flat, casos)):
    I_tabla, P_tabla = curva_PI(datos)
    ax.plot(I_grid, potencia(I_grid, I_tabla, P_tabla), label="P(I) interpolada")
    ax.plot(datos[:, 0], datos[:, 2], ".", ms=3, alpha=0.4, label="muestras del .npy")

    picos, _ = find_peaks(P_tabla, prominence=1.0)   # findpeaks como en la tarea. ignora peaks de menos de 1W
    ax.plot(I_tabla[picos], P_tabla[picos], "o", mfc="none", ms=9, label="maximos locales")
    print(f"caso {k}: maximos locales en I = {np.round(I_tabla[picos], 2)} A, "
          f"P = {np.round(P_tabla[picos], 1)} W")

    ax.set_title(f"caso {k}: {len(picos)} maximos locales")
    ax.set_ylabel("potencia (W)")
    ax.grid(alpha=0.3)

for ax in axs[1]:
    ax.set_xlabel("corriente (A)")
axs[0, 0].legend()
fig.tight_layout()
plt.show()