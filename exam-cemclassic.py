import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
from environment import Env, datos
import csv
from pathlib import Path

#1 seeds para reproducir
SEED = 42 
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
alpha = 0.25
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
historial_csv = []
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

        historial_csv.append({
            "iteracion": it,
            "evaluacion": evaluaciones,
            "caso": info["caso"],
            "I(Corriente)": info["I"],
            "P(Potencia)": float(reward),
            "mu(media)": float(mu.item()),
            "sigma(desviacion)": float(sigma.item()),
        })
    
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

    if valores_elite.max().item() <= 0.0:
        mu = torch.tensor((i_min + i_max)/ 2.0, dtype = torch.float32)
        sigma = torch.tensor((i_max - i_min) /3.0, dtype = torch.float32)
    else:
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

        if curr >0 and mejora_rel < tol_paro and sigma.item() <= sigma_min:
            print("Criterio de paro activado en iteracion: " + str(it))
            print("Mejora relativa : " + str(mejora_rel))
            print("Evaluaciones: " + str(evaluaciones))
            break

print("Entrenamiento finalizado, con " + str(evaluaciones) + " evaluaciones")
print("mu final: " + str(mu.item()))

# 5. Exportar a csv
carpeta_resultados = Path(__file__).resolve().parent / "resultados"
carpeta_resultados.mkdir(parents = True, exist_ok = True)
ruta_csv = carpeta_resultados / "historial_cem_classic.csv"

if historial_csv:
    with ruta_csv.open("w", newline = "", encoding = "utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames = historial_csv[0].keys())
        escritor.writeheader()
        escritor.writerows(historial_csv)
    print("Historial guiardado en: " + str(ruta_csv))

#6. gif

def gen_gif_cem(datos, registro_gif, carpeta_salida, nombre_archivo = "cem_classico.gif", fps = 6, cada = 1):
    n = len(registro_gif)
    if n == 0:
        print("No hay datos existentes")
        return
    
    iters = np.array([r["iter"] for r in registro_gif])
    casos = np.array([r["caso"] for r in registro_gif])
    mus = np.array([r["mu"] for r in registro_gif])
    sigmas = np.array([r["sigma"] for r in registro_gif])
    pots_elite = np.array([r["pot_elite"] for r in registro_gif])

    fig, (ax_curva, ax_panel) = plt.subplots(1, 2, figsize = (13,5))
    ax_pot = ax_panel.twinx()

    def draw(idx):
        ax_curva.clear()
        ax_panel.clear()
        ax_pot.clear()

        reg = registro_gif[idx]
        caso_act = int(reg["caso"])

        curva = datos[caso_act]
        orden = np.argsort(curva[:, 0])
        ax_curva.plot(curva[orden, 0], curva[orden, 2], color="blue", lw=2, label="Curva P-I activa")

        ax_curva.scatter(
            reg["muestras_I"],
            reg["muestras_p"],
            color="red",
            zorder=5,
            label="Muestras evaluadas (10)",
        )

        ax_curva.axvline(
            reg["mu"],
            color="green",
            ls="--",
            lw=2,
            label="Salida modelo μ = " + str(round(reg["mu"], 2)) + " A",
        )
        ax_curva.axvspan(
            reg["mu"] - reg["sigma"],
            reg["mu"] + reg["sigma"],
            color="green",
            alpha=0.15,
            label="Distribución μ ± σ",
        )
        
        ax_curva.set_xlabel("Corriente I (A)")
        ax_curva.set_ylabel("Potencia P (W)")
        ax_curva.set_xlim(i_min - 0.2, i_max + 0.2)
        ax_curva.set_ylim(-5, 220)
        ax_curva.legend(loc="upper left", fontsize=8)
        ax_curva.grid(alpha=0.3)

        l1 = ax_panel.plot(iters[: idx + 1], mus[: idx + 1], color="green", lw=2, label="Media μ (A)")
        l2 = ax_panel.plot(iters[: idx + 1], sigmas[: idx + 1], color="orange", lw=2, ls="--", label="Desv. Est. σ (A)")
        ax_panel.set_xlabel("Step")
        ax_panel.set_ylabel("Parámetros")
        ax_panel.set_xlim(0, max(1, n - 1))
        ax_panel.set_ylim(0, max(i_max, sigmas.max() + 0.5))
        ax_panel.grid(alpha=0.3)

        l3 = ax_pot.plot(
            iters[: idx + 1],
            pots_elite[: idx + 1],
            color="blue",
            lw=1.8,
            alpha=0.85,
            label="Potencia Media Élite (W)",
        )
        ax_pot.set_ylabel("Potencia Élite (W)", color="blue")
        ax_pot.set_ylim(0, max(220, pots_elite.max() + 15))

        lineas = l1 + l2 + l3
        etiquetas = [l.get_label() for l in lineas]
        ax_panel.legend(lineas, etiquetas, loc="lower right", fontsize=8)
        ax_panel.set_title("Evolucion de mu y sigma, desempeño", fontweight="bold")

    plt.tight_layout()
    frames = range(0, n, cada)
    anim = FuncAnimation(fig, draw, frames = frames, interval = 1000/fps)

    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents = True, exist_ok = True)
    ruta_gif = carpeta_salida / nombre_archivo
    anim.save(ruta_gif, writer = PillowWriter(fps = fps))
    plt.close(fig)
    print("Gif guardado en: " + str(ruta_gif.resolve()))

gen_gif_cem(
    datos=datos,
    registro_gif=registro_gif,
    carpeta_salida=str(carpeta_resultados),
    nombre_archivo="cem_clasico.gif",
    fps=6,
    cada=1,
)

"""Crear el ambiente en gymnasium -> cargo los parametros al ambiente que tiene
->Implemento Cross Entropy Method->creo el bucle, que va a ser:
1. Hago una distribucion gaussiana de la corriente
2. Se muestrean 10 valores de la distribución
3. El modelo evalua cada muestra en el entorno para dar la potencia 
4. Se ordenan las muestras por potencia, de mayor a menor
5. Actualiza mu y sigma
->Creo la visualización y grabación de GIF -> Entreno el modelo"""