import numpy as np
from collections import deque
from environment import Env, datos

from pathlib import Path
import csv
import torch 

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
    def __init__(self, media=4.0, sigma=1.5, alpha_actor=0.01, alpha_critic=0.01,
                 I_min=0.0, I_max=10.0, sigma_min=0.05, sigma_max=3.0,
                 sigma_reabrir=1.5, caida=0.3):
        self.media = media
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic

        #critic
        self.V = 0.0

        self.log_sigma = np.log(sigma) # aprendemos log(sigma) porque sigma es muy sensible cuando se hace pequeña:
                                        # el gradiente trae 1/sigma^3 y se dispara, y un paso grande podia dejarla negativa

        # limites de la politica
        self.I_min = I_min
        self.I_max = I_max
        self.log_sigma_min = np.log(sigma_min)
        self.log_sigma_max = np.log(sigma_max)

        # reabrir sigma si la potencia cae mucho cuando ya casi no explora
        self.sigma_reabrir = sigma_reabrir
        self.caida = caida 
        self.cambio = False

        # GIF metricas
        self.grad_media_ultimo = 0.0
        self.grad_sigma_ultimo = 0.0

    @property
    def sigma(self):
        return np.exp(self.log_sigma) # para que si se resta por un numero grande se vuelva positivo y no truene np.random.normal 

    def actuar(self):
        # muestreo con torch: I = media + sigma * epsilon; no cambia nada de usar torch a usar np.random.normal
        return float(self.media + self.sigma * torch.randn(1).item())

    def reabrir_exploracion(self):
        # sube sigma para buscar el pico del caso nuevo
        self.log_sigma = float(np.clip(np.log(self.sigma_reabrir),
                                       self.log_sigma_min, self.log_sigma_max))

    def detectar_cambio(self, advantage):
        # la potencia cayo respecto a lo que venia dando y ya casi no explora
        self.cambio = advantage < -self.caida and self.sigma < 0.3
        if self.cambio:
            self.reabrir_exploracion()
            self.V = 0.0

    def actualizar(self, distribucion, reward):
        advantage = reward - self.V

        gradiente_media = (distribucion - self.media) / self.sigma**2
        # log simga; tiene una derivada mas estable que sigma directamente
        # gradiente_sigma = ((distribucion - self.media)**2 / self.sigma**3) - (1 / self.sigma), si aumento el alpha con esta derivada el programa truena porque sigma se vuelve negativo
        # Derivada de log(sigma) respecto a sigma
        # derivda del logaritmo de la densidad de la politica = (a - mu)^2 / sigma^2 -1
        gradiente_sigma_log = ((distribucion - self.media) **2 / self.sigma**2 ) -1

        self.grad_media_ultimo = gradiente_media * advantage
        self.grad_sigma_ultimo = gradiente_sigma_log * advantage

        # Actualizacion del Actor - politica
        # (el paso de la media se multiplica por sigma**2 para cancelar el 1/sigma**2 y que no brinque cuando sigma es chica)
        self.media = self.media + self.alpha_actor * self.sigma**2 * gradiente_media * advantage
        self.media = float(np.clip(self.media, self.I_min, self.I_max))
        self.log_sigma = self.log_sigma + self.alpha_actor * advantage * gradiente_sigma_log
        self.log_sigma = float(np.clip(self.log_sigma, self.log_sigma_min, self.log_sigma_max))

        # Actualizacion del Critic
        self.V += self.alpha_critic * advantage

        self.detectar_cambio(advantage)
        return advantage


historial = []
#if __name__ == "__main__": 
env_mppt = Env(datos, steps_caso=100, max_steps=1500)
obs, info = env_mppt.reset(seed=SEED)
I_max = env_mppt.I_max
agente = ActorCrtic(media=1.2, sigma=0.5, alpha_actor=0.05, alpha_critic=0.1,
                    I_min=0.0, I_max=I_max)

# corrientes al azar para fijar la escala con potencia observada
P_warm = []
for _ in range(30):
    I_azar = torch.rand(1).item() * I_max
    _, r, _, _, _ = env_mppt.step(np.array([I_azar]))
    P_warm.append(r)
P_local_max = max(max(P_warm), 1e-8) 

evaluaciones = len(P_warm)   
recompensa_acumulada = 0.0

# criterio de paro, si la politica ya no explora y ya no se mueve
ventana, sigma_conv, tol_media = 30, 0.1, 0.05
medias_recientes = deque(maxlen=ventana)
sigmas_recientes = deque(maxlen=ventana)
parar_al_converger = False  

for step in range(1000):
    mu_antes = agente.media
    sigma_antes = agente.sigma
    V_antes = agente.V

    accion = agente.actuar()
    obs, reward, terminated, truncated, info = env_mppt.step(np.array([accion]))
    P_local_max = max(P_local_max, reward)
    advantage = agente.actualizar(accion, reward / P_local_max)

    evaluaciones += 1
    recompensa_acumulada += reward

    # el criterio se reinicia cuando se detecta un cambio de caso
    if agente.cambio:
        medias_recientes.clear()
        sigmas_recientes.clear()
    else:
        medias_recientes.append(mu_antes)
        sigmas_recientes.append(sigma_antes)
    convergido = (len(medias_recientes) == ventana
                  and max(sigmas_recientes) <= sigma_conv
                  and (max(medias_recientes) - min(medias_recientes)) < tol_media)

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
        'grad_media': agente.grad_media_ultimo,
        'grad_sigma': agente.grad_sigma_ultimo,
        'evaluaciones': evaluaciones,
        'recompensa_acumulada': recompensa_acumulada,
        'cambio_detectado': agente.cambio,
        'convergido': convergido,
            })

    if terminated or truncated:
        print("Entorno terminado")
        break

    if convergido and parar_al_converger:
        print(f"Convergio en {evaluaciones} evaluaciones")
        break

print(historial[0])

ruta_csv = Path(__file__).resolve().parent / "resultados" / "historial_actor_critic.csv"
ruta_csv.parent.mkdir(parents=True, exist_ok=True)

if historial:
    with ruta_csv.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=historial[0].keys())
        escritor.writeheader()
        escritor.writerows(historial)

    print(f"Historial guardado en: {ruta_csv}")