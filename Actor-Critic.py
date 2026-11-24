import numpy as np
import gymnasium as gym

np.random.seed(42)

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

ordenar_datos = np.argsort(I_datos)

I_datos = I_datos[ordenar_datos]
P_datos = P_datos[ordenar_datos]
#print(f'Corriente {I_datos[:10]}, Potencia {datos0[:10]}')

# aprende a donde conviene actuar  o sea mover la distribucion
media = 4
# aprende cuanto conviene explorar, para que en esos rangos si sucedio lejos un lugar con mucha potnecia podamos explorarlo
sigma = 1.5

V = 0 #Critic

#print("Corriente mínima:", I_datos.min())
#print("Corriente máxima:", I_datos.max())
#print("Potencia mínima:", P_datos.min())
#print("Potencia máxima:", P_datos.max())

#print("Primeros pares I-P:")
#print(np.column_stack((I_datos[:10], P_datos[:10])))

alpha = 0.01

for i in range(1000):
    escala_P = P_datos.max() 
    distribucion = np.random.normal(loc=media, scale=sigma)
    I = np.clip(distribucion, I_datos.min(), I_datos.max()) 
    P = np.interp(I, I_datos, P_datos)
    # Estandarizar el valor de recompensa
    reward = P / escala_P

    # Advantage
    advantage = reward - V

    gradiente_media = (distribucion - media) / sigma**2
    gradiente_sigma = ((distribucion - media)**2 / sigma**3) - (1 / sigma)

    # Actualizacion del Actor
    A = gradiente_media * (reward - V)

    
    media = media + alpha * gradiente_media * advantage

    

    sigma = sigma + alpha * advantage * gradiente_sigma

    # Actualización del Critic
    V = V + 0.01 * advantage
    
    if i % 100 == 0:
        print(
            f"paso {i}: I={I:.2f} "
            f"reward={reward} "
            f"V={V:.3f} "
            f"advantage={advantage:.3f} "
            f"media={media} "
            f"sigma={sigma:.3f}"
        )
    
    #print(f"paso {i}: I={I:.2f}  P={P:.2f}  P_norm={P/escala_P:.3f}")
    #print(f"gradiente: {gradiente}")



