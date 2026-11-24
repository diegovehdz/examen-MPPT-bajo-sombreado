import numpy as np
import torch 

datos0 = np.load("datasets/caso_0.npy")
datos1 = np.load("datasets/caso_1.npy")
datos2 = np.load("datasets/caso_2.npy")
datos3 = np.load("datasets/caso_3.npy")

def ppo(media_0, var_0, v_0):
    step = 300
    for i in range (0, step):

