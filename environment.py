from typing import Optional
import numpy as np
import gymnasium as gym
import torch
from gymnasium import spaces

print(torch.cuda.is_available())  # Debe devolver True si detecta la GPU y tiene CUDA
print(torch.version.cuda)         # Muestra la versión de CUDA incluida en tu PyTorch (ej. '12.1', '12.4')

np.random.seed(42)

class Env(gym.Env):

    def __init__(self, datos, max_steps: int | None = None, **kwargs, ):

        super().__init__(
            max_episode_steps = 100
            **kwargs,
        )
        
        # lista de los 4 casos
        self.datos = datos

        self.max_steps = max_steps

        # El espacio es continuo - https://gymnasium.farama.org/api/spaces/
        self.observation_space = spaces.Box(0, P_scale, shape=(1,)) #
        self.action_space = spaces.Box(0, I_max_local, shape(1,))

        

    def reset(self,):
        indice_r = self.np_random.integers(0, len(self.datos))
        set_datos = self.datos[indice_r]

        I_datos = set_datos[:,0]
        P_datos = set_datos[:,2]

        I_inicial = 4.2 #número arbitario 
        #ordenar lo datos
        ordenar_datos = np.argsort(I_datos)
        I_datos = I_datos[ordenar_datos]
        P_datos = P_datos[ordenar_datos]
        P_evaluado = np.interp(I_inicial, I_datos, P_datos)
        return (P_evaluado), indice_r

    def step(self, action):
        I = np.clip(action, I_min, I_max)
        P = self.potencia(I, self.datos[:, 0], self.datos[:, 2])
        reward = P * dt
        return obs, reward, terminated, truncated

    def procesar_datos(self, datos):
        for k, datos in enumerate(casos):
            I, V, P = datos[:, 0], datos[:, 1], datos[:, 2]
    return self.datos
    

    def curva_PI(self, datos):
        I, P = datos[:, 0], datos[:, 2]
        I_unica, idx = np.unique(I, return_inverse=True)           # ordenar y juntar repetidos
        P_unica = np.bincount(idx, weights=P) / np.bincount(idx)   # promediar los repetidos
        return np.concatenate([[0.0], I_unica]), np.concatenate([[0.0], P_unica])

    def potencia(self, I_consulta, I_tabla, P_tabla):
        # right = 0.0 significa que si pides más corriente que I_sc entonces el panel no entrega potencia
        # gracias google por
        return np.interp(I_consulta, I_tabla, P_tabla, right=0.0)

    def get_sate(self, obs):
        return None

    def get_action(self, state):
        return self.potencia(state, self.datos[:,0], self.datos[:,2])


        
