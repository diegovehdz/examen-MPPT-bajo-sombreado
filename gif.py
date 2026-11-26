import matplotlib
import matplotlib.pyplot as plt
import PIL
import numpy as np

# from exploracion import curva_PI

from environment import datos
from actor_critic import historial

def gen_gif(datos, historial, salida, fpd, cada, muestras, nombre_salida):
    n = len(historial)

    #obtener los campos del historial, no funciono la lambda function xd
    def campo(clave):
        return np.asarray([registro[clave] for registro in historial])
    
    caso, I, P = campo('caso'), campo('I(Corriente)'), campo('P(Potencia)') #dividir en listas cada campo con sus datos
    media, sigma = campo('mu(media)'), campo('sigma(desviacion)')
    V, adv = campo('V(critic)'), campo('advantage')

    fig, (ax_curva, ax_metricas) = plt.subplots(1, 2, figsize=(12, 5))

    indice = -1
    caso_actual = int(caso[indice])

    curva = datos[caso_actual]
    orden = np.argsort(curva[:,0])
    I_curva = curva[orden, 0]
    P_curva = curva[orden, 2]

    ax_curva.plot(I_curva, P_curva, label="Curva P-I")
    ax_curva.scatter(I[indice], P[indice], color="red", label="Muestra evaluada")
    ax_curva.set(xlabel="Corriente I (A)", ylabel="Potencia P (W)")
    ax_curva.legend()

    pasos = np.arange(len(V))

    ax_metricas.plot(pasos, V, marker="o", label="Crítico V")
    ax_metricas.plot(pasos, adv, marker=".", label="Advantage")
    ax_metricas.set_xlabel("Paso")
    ax_metricas.set_ylabel("Valor")
    ax_metricas.set_title("Crítico y ventaja")
    ax_metricas.grid(alpha=0.3)
    ax_metricas.legend()

    plt.tight_layout()
    plt.show()

if __name__ == """__main__""":
    print("registros en historial:", len(historial))
    gen_gif(datos, historial=historial, salida='resultados',fpd=8, cada=5, muestras=10, nombre_salida='actor_critic.gif')
    
