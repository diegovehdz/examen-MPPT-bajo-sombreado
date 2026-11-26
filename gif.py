import matplotlib
import matplotlib.pyplot as plt
import PIL
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter #https://matplotlib.org/stable/api/animation_api.html
import os

# from exploracion import curva_PI

from environment import datos
from actor_critic import historial


def gen_gif(datos, historial, salida, fpd, cada, muestras, nombre_salida):
    n = len(historial)

    def campo(clave):
        return np.asarray([r[clave] for r in historial])

    caso, I, P = campo('caso'), campo('I(Corriente)'), campo('P(Potencia)')
    media, sigma = campo('mu(media)'), campo('sigma(desviacion)')
    V, adv = campo('V(critic)'), campo('advantage')

    fig, (ax_curva, ax_metricas) = plt.subplots(1, 2, figsize=(12, 5))
    ymin, ymax = min(V.min(), adv.min()), max(V.max(), adv.max())

    def dibujar(indice):
        ax_curva.clear()
        ax_metricas.clear()

        caso_actual = int(caso[indice])
        curva = datos[caso_actual]
        orden = np.argsort(curva[:, 0])
        ax_curva.plot(curva[orden, 0], curva[orden, 2], label="Curva P-I")

        mismo_caso = np.where(caso[:indice + 1] == caso_actual)[0][-muestras:]
        ax_curva.scatter(I[mismo_caso], P[mismo_caso], color="red",
                         label="Muestras evaluadas")
        ax_curva.axvline(media[indice], color="green", ls="--", label="μ")
        ax_curva.axvspan(media[indice] - sigma[indice],
                         media[indice] + sigma[indice],
                         color="green", alpha=0.15, label="μ (+ -) σ")
        ax_curva.set(xlabel="Corriente I (A)", ylabel="Potencia P (W)",
                     title=f"Caso {caso_actual} | Paso {indice}")
        ax_curva.legend(); ax_curva.grid(alpha=0.3)

        pasos = np.arange(indice + 1)
        ax_metricas.plot(pasos, V[:indice + 1], label="Crítico V")
        ax_metricas.plot(pasos, adv[:indice + 1], label="Advantage")
        ax_metricas.set(xlabel="Paso", ylabel="Valor", title="Crítico y ventaja",
                        xlim=(0, n), ylim=(ymin, ymax))
        ax_metricas.legend(); ax_metricas.grid(alpha=0.3)

    plt.tight_layout()
    anim = FuncAnimation(fig, dibujar, frames=range(0, n, cada),
                         interval=1000 / fpd)

    os.makedirs(salida, exist_ok=True)
    anim.save(os.path.join(salida, nombre_salida), writer=PillowWriter(fps=fpd))
    plt.close(fig)

if __name__ == """__main__""":
    print("registros en historial:", len(historial))
    gen_gif(datos, historial=historial, salida='resultados',fpd=8, cada=2, muestras=10, nombre_salida='actor_critic.gif')
    
