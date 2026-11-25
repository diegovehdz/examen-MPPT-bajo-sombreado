import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import os
from environment import Env, datos

#1 seeds para reproducir
SEED = 67 #Seed de prueba, cambiar a 42 para el final
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#2 Inicializar env
env = Env(datos, steps_caso = 200, max_steps = 3000)

obs, info = env.reset(seed = SEED)
i_min = float(env.action_space.low[0])
i_max = float(env.action_space.high[0])

print("Entorno ok")
print("Caso inicial: " + str(info["caso"]))
print("I min : " + str(i_min))
print("I max : " + str(i_max))
print("Potencia inicial: " + str(obs[0]))

"""Crear el ambiente en gymnasium -> cargo los parametros al ambiente que tiene
->Implemento Cross Entropy Method->creo el bucle, que va a ser:
1. Hago una distribucion gaussiana de la corriente
2. Se muestrean 10 valores de la distribución
3. El modelo evalua cada muestra en el entorno para dar la potencia 
4. Se ordenan las muestras por potencia, de mayor a menor
5. Actualiza mu y sigma
->Creo la visualización y grabación de GIF -> Entreno el modelo"""