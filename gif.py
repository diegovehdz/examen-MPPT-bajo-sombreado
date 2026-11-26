import matplotlib
import matplotlib.pyplot as plt
import PIL
import numpy as np

from exploracion import curva_PI

def gen_gif(datos, historial, salida, fpd, cada, muestras, nombre_salida):
    n = len(historial)

    #obtener los campos del historial
    campo = lambda k: np.array(h[k] for h in historial)
    caso, I, P = campo('caso'), campo('I'), campo('P') #dividir en listas cada campo con sus datos
    media, sigma = campo('mu(media)'), campo('sigma(desviacion)')
    V, adv = campo('V(critic)'), campo('advantage')
    
