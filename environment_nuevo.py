"""
las propuestas que les puse chat:
A. acción: 
- opción 1: la corriente absoluta, entre 0 y 8.9A (el max que se ve)
- opción 2: es un incremento delta_I, o sea si sube o baja la corriente 
- yo digo que la absoluta porque la potencial solo depende de la corriente que se elige y cada paso sería independiente, se parece más a lo del profe
B. observación: la potencia media, este no cambia
C. recompensa: P / P_ref, con P_ref fija
- la propuesta sería que P_ref = I_sc_max × V_oc_max ≈ 8.9 × 37 ≈ 329 W. ningún panel entrega más que I_sc x V_oc y no usa el máximo real de ninguna curva
- como en el caso 2 da como mucho 0.14 y el caso 0 da 0.61 hay que normalizar las recompensas dentro de cada batch de muestras
- rolen más ideas :v
D. cambio de caso: un parámetro steps_by_case o algo así
- una variable para jugar : cada pasos_por_caso pasos, el entorno elige al azar otro de los 4 casos
- el random sale de una seed obvs
E. fin del episodio:
- 'terminated' siempre es False. terminar al "llegar al máximo" requeriría conocer el máximo real y se supone que eso está prohibido xd
- puede ser que 'truncated' pasa a True cuando se llega a max_pasos u otra variable que podamos definir, no?
F. info: solo para gráficas. tendrá {"caso": k, "cambio_caso": True/False}. igual y el gif lo necesita para marcar el cambio de caso, pero el agente nunca debe leerlo
G. una step es una evaluación. si por ejemplo el REINFORCE que me tocó muestrea 10 corrientes por actualización, hace 10 llamadas a step,  así la métrica de número de evaluaciones sale directa

un pseudo código del step se vería masomenos así
step(I):
    I = recortar(I, 0, 8.9)
    P = interpolar(I, tabla del caso activo)    # funcion potencia()
    step+= 1
    si step es multiplo de step_by_case:
        caso = otro caso al azar
    regresar [P/P_ref], P/P_ref, False, step>= max_steps, info
----------------------------------------------------------------------------------------------------------------------
el agente propone una corriente y el entorno regresa la potencia que entrega
el panel con esa corriente. El caso de sombreado activo cambia al azar cada
'pasos_por_caso' pasos. 
!!!! OJO: el agente nunca conoce el caso ni el máximo real que nosotros sí vimos en el script de exploracion !!!!!!
"""
from pathlib import Path

import numpy as np
import gymnasium as gym

CARPETA_DATOS = Path(__file__).resolve().parent / "datasets"
N_CASOS = 4


# acá hacemos la interpolación de la tabla para poder consultar la potencia a cualquier corriente
# con un promedio de los puntos repetidos
def curva_PI(datos):
    I, P = datos[:, 0], datos[:, 2]
    I_unica, idx = np.unique(I, return_inverse=True)           # ordenar y juntar repetidos
    P_unica = np.bincount(idx, weights=P) / np.bincount(idx)   # promediar los repetidos
    
    # agregar el punto (0 A, 0 W)
    # en circuito abierto no hay potencia lol
    return np.concatenate([[0.0], I_unica]), np.concatenate([[0.0], P_unica])


def potencia(I_consulta, I_tabla, P_tabla):
    # right = 0.0 significa que si pides más corriente que I_sc entonces el panel no entrega potencia
    # gracias google por
    return np.interp(I_consulta, I_tabla, P_tabla, right=0.0)


def cargar_casos(carpeta=CARPETA_DATOS):
    """lee los 4 .npy y regresa (tablas, I_max, V_max).

    tablas: lista con un par (I_tabla, P_tabla) por caso.
    I_max:  la corriente de corto circuito más grande -> límite de la acción.
    V_max:  el voltaje de circuito abierto más grande -> para la potencia de referencia.
    """
    datos = [np.load(Path(carpeta) / f"caso_{k}.npy") for k in range(N_CASOS)]
    tablas = [curva_PI(d) for d in datos]
    I_max = max(d[:, 0].max() for d in datos)
    V_max = max(d[:, 1].max() for d in datos)
    return tablas, I_max, V_max