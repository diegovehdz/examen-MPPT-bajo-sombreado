from typing import Optional
import numpy as np
import gymnasium as gym
import torch

print(torch.cuda.is_available())  # Debe devolver True si detecta la GPU y tiene CUDA
print(torch.version.cuda)         # Muestra la versión de CUDA incluida en tu PyTorch (ej. '12.1', '12.4')

class Env(gym.Env):

    def __init__(self, datos):

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        
        self.datos = datos

        def procesar_datos(self, datos):
            for k, datos in enumerate(casos):
                I, V, P = datos[:, 0], datos[:, 1], datos[:, 2]
        return datos
        
    
        def curva_PI(self, datos):
            I, P = datos[:, 0], datos[:, 2]
            I_unica, idx = np.unique(I, return_inverse=True)           # ordenar y juntar repetidos
            P_unica = np.bincount(idx, weights=P) / np.bincount(idx)   # promediar los repetidos
            return np.concatenate([[0.0], I_unica]), np.concatenate([[0.0], P_unica])

        def potencia(I_consulta, I_tabla, P_tabla):
            # right = 0.0 significa que si pides más corriente que I_sc entonces el panel no entrega potencia
            # gracias google por
            return np.interp(I_consulta, I_tabla, P_tabla, right=0.0)

        def reset(self, **kwargs):
            obs = super().reset(**kwards)
            return obs
        
        def step(self, action):
            obs, reward,terminated, truncated, info = self.env.step(action)
            return obs, reward, terminated, truncated, info

        
