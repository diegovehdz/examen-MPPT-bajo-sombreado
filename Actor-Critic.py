import numpy as np
import gymnasium as gym

from environment import Env, datos

SEED = 42
np.random.seed(SEED)

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
"""
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
        print(P)
    #print(f"paso {i}: I={I:.2f}  P={P:.2f}  P_norm={P/escala_P:.3f}")
    #print(f"gradiente: {gradiente}")
"""

class ActorCrtic:
    def __init__(self, media=4.0, sigma=1.5, alpha_actor=0.01, alpha_critic=0.01):
        self.media = media
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic

        #critic
        self.V = 0.0

        self.log_sigma = np.log(sigma) # aprendemos log(sigma) porque sigma es muy sensible cuando se hace pequeña:
                                        # el gradiente trae 1/sigma^3 y se dispara, y un paso grande podia dejarla negativa
    
    @property
    def sigma(self):
        return np.exp(self.log_sigma) # para que si se resta por un numero grande se vuelva positivo y no truene np.random.normal 

    def actuar(self):
        #muestrar la corriente con se hizo en las pruebas
        return np.random.normal(self.media, self.sigma)

    def actualizar(self, distribucion, reward):
        advantage = reward - self.V

        gradiente_media = (distribucion - self.media) / self.sigma**2
        # log simga; tiene una derivada mas estable que sigma directamente
        # gradiente_sigma = ((distribucion - self.media)**2 / self.sigma**3) - (1 / self.sigma), si aumento el alpha con esta derivada el programa truena porque sigma se vuelve negativo
        # Derivada de log(sigma) respecto a sigma
        # derivda del logaritmo de la densidad de la politica = (a - mu)^2 / sigma^2 -1
        gradiente_sigma_log = ((distribucion - self.media) **2 / self.sigma**2 ) -1

        # Actualizacion del Actor - politica
        self.media = self.media + self.alpha_actor * gradiente_media * advantage
        self.log_sigma = self.log_sigma + self.alpha_actor * advantage * gradiente_sigma_log

        # Actualizacion del Critic
        self.V += self.alpha_critic * advantage
        return advantage


historial = []
if __name__ == "__main__": 
    env_mppt = Env(datos, steps_caso=50, max_steps=1000)
    obs, info = env_mppt.reset(seed=SEED)
    agente = ActorCrtic(media=3.8, sigma=0.2, alpha_actor=0.09, alpha_critic=0.09)
    P_local_max = 1e-8 # un valor extremadamente bajo para que lo cambie por el maximo local que se vea

    for step in range(1000):
        mu_antes = agente.media
        sigma_antes = agente.sigma
        V_antes = agente.V

        accion = agente.actuar()
        obs, reward, terminated, truncated, info = env_mppt.step(np.array([accion]))
        P_local_max = max(P_local_max, reward)
        advantage = agente.actualizar(accion, reward / P_local_max)

        historial.append({
            'step':step,
            'caso':info["caso"],
            'I(Corriente)': info["I"],
            'P(Potencia)': reward,
            'reward_norm': (reward / P_local_max),
            'mu(media)': mu_antes,
            'sigma(desviacion)':sigma_antes,
            'V(critic)': V_antes,
            'advantage': advantage,
                })
        
        if step % 20 == 0:
            print(f"Paso: {step}, Caso: {info['caso']}, I: {info['I']:.2f}, "
                  f"P: {reward:.1f}, media: {agente.media:.2f}, "
                  f"sigma: {agente.sigma:.3f}, V: {agente.V:.3f}, adv: {advantage:.3f}")

        

        if terminated or truncated:
            print("Entorno terminado")
            break

    print(historial[0])
