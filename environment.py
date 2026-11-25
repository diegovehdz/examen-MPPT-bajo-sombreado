from typing import Optional
import numpy as np
import gymnasium as gym
import torch
from gymnasium import spaces

print(torch.cuda.is_available())  # Debe devolver True si detecta la GPU y tiene CUDA
print(torch.version.cuda)         # Muestra la versión de CUDA incluida en tu PyTorch (ej. '12.1', '12.4')

np.random.seed(42)


datos0 = np.load("datasets/caso_0.npy")
datos1 = np.load("datasets/caso_1.npy")
datos2 = np.load("datasets/caso_2.npy")
datos3 = np.load("datasets/caso_3.npy")

datos = [datos0, datos1, datos2, datos3]


class Env(gym.Env):

    def __init__(self, datos, steps_caso = 10,max_steps = 100,):

        super().__init__(  )
        
        # lista de los 4 casos
        self.datos = datos

        self.max_steps = max_steps
        self.steps_caso =  steps_caso

        self.I_casos, self.P_casos = [], []

        for caso in datos:
            orden =  np.argsort(caso[:, 0])
            self.I_casos.append(caso[orden, 0])
            self.P_casos.append(caso[orden, 2])

        # encontrar el espacio de accion, encontrar el espacio de observacion en el cual vamos a trabajar
        self.I_max, self.I_min = 0, 0
        for i in datos:
            I, V, P = i[:, 0], i[:, 1], i[:, 2]
            self.I_max = max(self.I_max,np.max(I))
            self.I_min = min(self.I_min, np.min(I))


        # El espacio es continuo - https://gymnasium.farama.org/api/spaces/
        self.observation_space = spaces.Box(0, 10000, shape=(1,)) #
        self.action_space = spaces.Box(self.I_min, self.I_max, shape=(1,))


        self.caso_actual = None
        
        #contar los pasos hasta llegar a cambiar de dataset
        self.t_contador = 0

        

    def reset(self, *, seed =None):
        super().reset(seed = seed)
        indice_r = self.np_random.integers(0, len(self.datos))
        indice_r = int(indice_r)
        self.caso_actual = indice_r

        self.t_contador = 0
        I_inical = 4.2 #numero arbitrario

        P_evaluado = self.potencia(I_inical)
        return np.array([P_evaluado], dtype=np.float32), {"caso": indice_r}

    def step(self, action):
        self.t_contador += 1
        if self.t_contador % self.steps_caso == 0:
            self.caso_actual = int(self.np_random.integers(0, len(self.datos)))

        I = float(np.clip(np.asarray(action).ravel()[0], self.I_min, self.I_max)) # me daba error antes porque era un arreglo, y ocupaba un escalar para poder evaluarlo
        P = self.potencia(I)
        obs = np.array([P], dtype=np.float32)
        truncated = self.t_contador >= self.max_steps

        return obs, float(P), False, truncated, {"caso": self.caso_actual, "I": I}
    

    def potencia(self, I):
        # right = 0.0 significa que si pides más corriente que I_sc entonces el panel no entrega potencia
        # gracias google por
        return np.interp(I, self.I_casos[self.caso_actual], self.P_casos[self.caso_actual], right=0.0)

    
    """
    def get_sate(self, obs):
        return None

    la accion es potencia func 

    def get_action(self, state):
        return self.potencia(state, self.datos[:,0], self.datos[:,2])
    """

if __init__ == "__main__":
    env = Env(datos, steps_caso=50)
    obs, info = env.reset(seed =42)
    caso_prev = info["caso"]
    for _ in range(120):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if info["caso"] != caso_prev:
            print("cambio a caso", info["caso"])
            caso_prev = info["caso"]


            
