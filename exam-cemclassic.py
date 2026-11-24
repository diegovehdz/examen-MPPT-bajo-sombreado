import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

data_path_np = "./examen-MPPT-bajo-sombreado/datasets/caso_"
num_case = 4

casos = []

for i in range(num_case):
    data + str(i) = np.load(data_path_np + str(i) + ".npy")


class env_caso(gym.env):
    def  __init__(self, casos, steps = 30, max_steps = 300):
        super().__init__()

        self.action_spce  = gym.spaces.Box(
            low = np.array([i_min], dtype = np.float32),
            high = np.array([i_max], dtype = np.float32),
            dtype = np.float32
        )

"""Crear el ambiente en gymnasium -> cargo los parametros al ambiente que tiene
->Implemento Cross Entropy Method->creo el bucle, que va a ser:
1. Hago una distribucion gaussiana de la corriente
2. Se muestrean 10 valores de la distribución
3. El modelo evalua cada muestra en el entorno para dar la potencia 
4. Se ordenan las muestras por potencia, de mayor a menor
5. Actualiza mu y sigma
->Creo la visualización y grabación de GIF -> Entreno el modelo"""