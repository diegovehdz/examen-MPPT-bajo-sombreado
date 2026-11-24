import numpy as np
import gymnasium as gym

# Importar los datos
datos0 = np.load("datasets/caso_0.npy")
datos1 = np.load("datasets/caso_1.npy")
datos2 = np.load("datasets/caso_2.npy")
datos3 = np.load("datasets/caso_3.npy")

# print("Primer elemento:", datos0[0])

# Actor 
class ActorCritic:
    def __init__(self, datos):
        self.datos = datos
    def actor()