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
env = Env(datos, steps_caso = 50, max_steps = 1000)

obs, info = env.reset(seed = SEED)
i_min = float(env.action_space.low[0])
i_max = float(env.action_space.high[0])


#3 hiperparametros
n_samples = 10
elite_frac = 0.2
n_elite = max(1, int(n_samples * elite_frac))
alpha = 0.5
sigma_min = 0.45
max_iter = 1000

#Parametros de stop 
ventana_paro = 15
tol_paro =  1e-2

#Initial dist
mu = torch.tensor((i_min + i_max)/ 2.0, dtype = torch.float32)
sigma = torch.tensor((i_max - i_min) /3.0, dtype = torch.float32)

historial_elite = []
registro_gif = []
evaluaciones = 0
caso_anterior = info["caso"]

print("Empieza cem")
print("Caso inicial: " + str(caso_anterior))
print("Mu" + str(mu.item()))
print("Sigma" + str(sigma.item()))

#4 Bucle Cem
for it in range(max_iter):
    #Muestrear
    dist = torch.distributions.Normal(mu, sigma)
    muestras = torch.clamp(dist.sample((n_samples,)), min = i_min, max = i_max)

    potencias = []
    casos_gen = []

    for i_val in muestras : 
        accion = np.array([i_val.item()], dtype = np.float32)
        obs, reward, terminated, truncated, info = env.step(accion)
        evaluaciones += 1

        potencias.append(float(reward))
        casos_gen.append(info["caso"])

        if info["caso"] != caso_anterior:
            print("Cambio de caso, de:" + str(caso_anterior) + "a" + str(info["caso"]))
            caso_anterior = info["caso"]

        if terminated or truncated:
            obs, info = env.reset()
            caso_anterior = info["caso"]


    potencias_t = torch.tensor(potencias, dtype = torch.float32)

    #Seleccionar el conjunto con mayor potencia
    valores_elite, idx_elite = torch.topk(potencias_t, k = n_elite)
    muestras_elite = muestras[idx_elite]

    #Actualizar mu y sigma 

    mu_elite = muestras_elite.mean()
    sigma_elite = muestras_elite.std(unbiased = False)


    mu = alpha * mu_elite + (1.0 - alpha) * mu
    sigma = torch.clamp(alpha * sigma_elite + (1.0 -alpha)* sigma, min = sigma_min)

    pot_media_elite = float(valores_elite.mean().item())
    historial_elite.append(pot_media_elite)

    #Guardar info

    registro_gif.append({
        "iter":it,
        "caso": casos_gen[-1],
        "mu": float(mu.item()),
        "sigma" : float(sigma.item()),
        "muestras_I" : muestras.tolist(),
        "muestras_p" : potencias,
        "pot_elite" : pot_media_elite
    })

    if it %10 == 0:
        print("Iteracion: " + str(it) + "Caso: " + str(casos_gen[-1]) )
        print("Mu:" + str(mu.item()))
        print("Sigma" + str(sigma.item()))
        print("Potencia media: " + str(pot_media_elite))


    #Criterio de paro no ground truth, se detiene si la mejora de media movil de pot elite es menor a 
    #tol paro  y sigma ya convergio cerca del minimo

    if len(historial_elite) >= 2 * ventana_paro:
        prev = sum(historial_elite[-2 * ventana_paro : -ventana_paro]) / ventana_paro
        curr = sum(historial_elite[-ventana_paro:])/ventana_paro
        mejora_rel = abs(curr- prev)/ (abs(prev) + 1e-8)

        if mejora_rel < tol_paro and sigma.item() <= sigma_min:
            print("Criterio de paro activado en iteracion: " + str(it))
            print("Mejora relativa : " + str(mejora_rel))
            print("Evaluaciones: " + str(evaluaciones))
            break

print("Entrenamiento finalizado, con " + str(evaluaciones) + " evaluaciones")
print("mu final: " + str(mu.item()))


"""Crear el ambiente en gymnasium -> cargo los parametros al ambiente que tiene
->Implemento Cross Entropy Method->creo el bucle, que va a ser:
1. Hago una distribucion gaussiana de la corriente
2. Se muestrean 10 valores de la distribución
3. El modelo evalua cada muestra en el entorno para dar la potencia 
4. Se ordenan las muestras por potencia, de mayor a menor
5. Actualiza mu y sigma
->Creo la visualización y grabación de GIF -> Entreno el modelo"""