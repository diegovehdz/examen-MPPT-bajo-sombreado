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

V = 0 #Critic

for i in range(30):
    escala_P = P_datos.max() 
    distribucion = np.random.normal(loc=media, scale=sigma)
    I = np.clip(distribucion, I_datos.min(), I_datos.max()) 
    P = np.interp(I, I_datos, P_datos)
    # Estandarizar el valor de recompensa
    reward = P / escala_P

    # Advantage
    advantage = reward - V

    gradiente = (I - media) / sigma**2
    A = gradiente * (reward - V)

    # Actualizacion del Actor
    media = media + 0.01 * gradiente * advantage

    sigma = sigma + 0.01 * A
    sigma = np.abs(sigma)

    advantage = reward - V

    # Actualización del Critic
    V = V + 0.01 * advantage
    
    if i % 5 == 0:
        print(
            f"paso {i}: I={I:.2f} "
            f"reward={reward:.3f} "
            f"V={V:.3f} "
            f"advantage={advantage:.3f} "
            f"media={media:.3f}"
        )
    
    #print(f"paso {i}: I={I:.2f}  P={P:.2f}  P_norm={P/escala_P:.3f}")
    #print(f"gradiente: {gradiente}")



