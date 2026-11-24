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