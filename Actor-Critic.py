import numpy as np
import gymnasium as gym

# Importar los datos
datos0 = np.load("datasets/caso_0.npy")
datos1 = np.load("datasets/caso_1.npy")
datos2 = np.load("datasets/caso_2.npy")
datos3 = np.load("datasets/caso_3.npy")

# print("Primer elemento:", datos0[0])
 

"""
inicializar actor random, tomar el valor de la Potencia de la interpolacion
a partir de ese valor se calcula el Critic
el critic le dice que tan buena o mala fue la accion
"""

# print(f'forma {datos0.shape}')
I_datos = datos0[:,0]
P_datos = datos0[:,2]

#print(f'Corriente {I_datos[:10]}, Potencia {datos0[:10]}')

media = 4
sigma = 1.5
rng = np.random.default_rng(42)
escala_P = P_datos.max()        
distribucion = np.random.normal(loc=media, scale=sigma, size=10)

V = 0 #Critic
A = 0 #Actor
for i in range(10):
    escala_P = P_datos.max() 
    distribucion = np.random.normal(loc=media, scale=sigma, size=10)
    I = np.clip(distribucion[i], I_datos.min(), I_datos.max()) 
    P = np.interp(I, I_datos, P_datos)
    # Estandarizar el valor de recompensa
    reward = P / escala_P

    gradiente = (I - media) / sigma**2
    V = V + 0.01 * (reward - V)
    A = A + 0.01 * gradiente * (reward - V)
    
    #print(f"paso {i}: I={I:.2f}  P={P:.2f}  P_norm={P/escala_P:.3f}")
    #print(f"gradiente: {gradiente}")



