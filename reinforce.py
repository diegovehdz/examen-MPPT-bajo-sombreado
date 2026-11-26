import math
import numpy as np
import torch
from environment import Env, datos
from collections import deque
import csv
import math
from pathlib import Path

CARPETA_RESULTADOS = Path(__file__).resolve().parent / "resultados"

class PoliticaGaussiana:
    """
    politica pi(I) = N(mu, sigma)
    - mu se guarda directo en amperes y se recorta a [0, I_max] en cada actualizacion
    - sigma es log_sigma para que siempre sea positiva: sigma = exp(log_sigma)
      y se recorta a [sigma_min, sigma_max]
    """

    def __init__(self, I_max, mu_inicial=None, sigma_inicial=1.0, sigma_min=0.1, sigma_max=3.0):
        self.I_max = float(I_max)
        self.mu = self.I_max / 2 if mu_inicial is None else float(mu_inicial)
        self.log_sigma = math.log(sigma_inicial)
        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_max = math.log(sigma_max)

    @property
    def sigma(self):
        return math.exp(self.log_sigma)

    def muestrear(self, n):
        # I = mu + sigma * epsilon
        return self.mu + self.sigma * torch.randn(n)
    
    def reabrir_exploracion(self, sigma_nueva):
        # es para buscar el pico del caso nuevo porque en las pruebas se queda atorado en un pico local
        # #hashtagputamadre
        self.log_sigma = min(max(math.log(sigma_nueva), self.log_sigma_min), self.log_sigma_max)


def evaluar(env, politica, n_muestras=10):
    """
    muestrea n corrientes de la politica y evalua cada una en el entorno

    cada muestra es 1 llamada al step,
    un step es una evaluacion de la curva

    regresa:
    - corrientes:  tensor con las corrientes que salieron de la politica (sin recortar)
    - recompensas: tensor con la W que regreso el entorno
    - info_lote:   SOLO para visualizar /// el agente no debe usarlo !!!!!!!!
    """
    corrientes = politica.muestrear(n_muestras)

    recompensas, casos, truncado = [], [], False
    for I in corrientes:
        # el entorno espera un arreglo de forma (1,) y recorta I a [0, I_max]
        _, r, _, truncated, info = env.step(np.array([I.item()], dtype=np.float32))
        recompensas.append(r)
        casos.append(info["caso"])
        truncado = truncado or truncated

    recompensas = torch.tensor(recompensas, dtype=torch.float32)
    return corrientes, recompensas, {"casos": casos, "truncado": truncado}


def actualizar_politica(politica, corrientes, recompensas, alpha_mu=0.3, alpha_sigma=0.1):
    """
    un paso de REINFORCE con baseline (el promedio del batch), todo a mano

    1. ventaja:   A_i = (r_i - promedio(r)) / desviacion(r)
    2. gradiente: derivado a mano de log pi (ver abajo)
    3. ascenso de gradiente con paso escalado (gradiente natural)

    regresa un dict con valores para metricas y para el GIF
    """
    mu, sigma = politica.mu, politica.sigma

    # 1. ventaja: que tan buena fue cada muestra comparada con las demas del mismo batch
    # +1e-8 evita 0/0 = NaN si todas las recompensas son iguales ??????????
    # si todas las recompensas son iguales, la ventaja seria 0/0 = nan error  
    ventajas = (recompensas - recompensas.mean()) / (recompensas.std() + 1e-8)

    # 2. gradiente de reinforce con baseline (promedio del batch)
    #    log pi(I)              = -log(sigma) - 0.5*log(2*pi) - (I - mu)^2 / (2*sigma^2)
    #    d log pi / d mu        = (I - mu) / sigma^2
    #    d log pi / d log_sigma = (I - mu)^2 / sigma^2 - 1
    #    gradiente = promedio( A_i * d log pi(I_i) )
    z = (corrientes - mu) / sigma  # distancia de cada muestra a mu, medida en sigmas
    grad_mu = (ventajas * z / sigma).mean().item()
    grad_log_sigma = (ventajas * (z ** 2 - 1)).mean().item()
    norma_grad = math.sqrt(grad_mu ** 2 + grad_log_sigma ** 2)

    # 3. ascenso de gradiente (suma porque queremos maximizar la potencia)
    #    - mu: el paso se multiplica por sigma^2 para cancelar el 1/sigma^2 del gradiente
    #      sin esto los pasos explotan cuando sigma es chica y mu brinca los escalones
    #    - log_sigma: el 0.5 viene de la misma idea de gradiente natural de una gaussiana
    paso_mu = alpha_mu * sigma ** 2 * grad_mu
    paso_log_sigma = alpha_sigma * 0.5 * grad_log_sigma

    politica.mu = min(max(mu + paso_mu, 0.0), politica.I_max)
    politica.log_sigma = min(max(politica.log_sigma + paso_log_sigma, politica.log_sigma_min), politica.log_sigma_max)

    return {
        "grad_mu": grad_mu,
        "grad_log_sigma": grad_log_sigma,
        "norma_grad": norma_grad,
        "paso_mu": paso_mu,
        "recompensa_media": recompensas.mean().item(),
    }

class DetectorCambio:
    """
    detectar un cambio de caso usando sola la potencia observada ( PEROnunca info["caso"])

    supongo que la manera más sencilla es con dos senales:
    - sin_info: todas las recompensas del batch son iguales o sea (todas 0W porque mu
     quedó pasando la I_sc del caso nuevo como en los casos 0 y 2) so el gradiente es 0 y la politica no se puede mover
    - "cambio: cuando una politica explota porque ya encontro un pico local, y si de repente la potencia media baja mucho 
     es porque el caso cambio y ahora la politica esta en un pico local distinto
     
     
     
     si la sigma de la politica es grande no hay que preocuparse porque 
     la politica todavia esta explorando y puede encontrar el pico local del caso nuevo
     pero si la sigma es chica y la potencia media baja mucho es porque 
     la politica se quedo atorada en un pico local del caso viejo como ya pasó en la seed 0 y la seed 67
    """

    def __init__(self, umbral=0.3, beta=0.3, sigma_vigilar=0.3):
        self.umbral = umbral                # un salto de mas de 30%, no es perfecto pero funciona para el dataset xd
        self.beta = beta                    # peso del batch nuevo en el promedio movil
        self.sigma_vigilar = sigma_vigilar  # solo vigilar saltos cuando sigma < eso
        self.P_ref = None                   # referencia de potencia media del batch anterior para detectar un salto

    def revisar(self, recompensas, sigma):
        P = recompensas.mean().item()

        if recompensas.std().item() < 1e-6:  # todas iguales: no hay de donde aprender
            self.P_ref = None
            return "sin_info"

        if self.P_ref is None:  # primer batch o justo despues de una deteccion
            self.P_ref = P
            return None

        salto = abs(P - self.P_ref) > self.umbral * self.P_ref
        if sigma < self.sigma_vigilar and salto:
            self.P_ref = None
            return "cambio"

        self.P_ref = (1 - self.beta) * self.P_ref + self.beta * P  # promedio de la potencia media del batch anterior y del batch actual
        return None

class CriterioParo:
    """
    para que converja el entrenamiento, la politica tiene que dejar de explorar cuando ya casi no explora
    la política convergió cuando en las últimas n actualizaciones
        - sigma se mantuvo chica. o sea ya casi no explora. uso 0.15 y no 0.1 porque sigma_min es 0.1 y no baja de ahí, 
        si se queda en 0.1A no hay forma de saber si ya convergió o no (sigma_conv = 0.15)
        - mu se movió menos de 0.05A entre su valor mas alto y el mas bajo o sea que el gradiente ya no la empuja a ningún lado (max - min < tol_mu)
        
    de todos modos no sabemos el pico global del caso, solo sabemos que la politica ya no se mueve y ya casi no explora
    """

    def __init__(self, n=10, sigma_conv=0.15, tol_mu=0.05):
        self.n = n
        self.sigma_conv = sigma_conv  # un poco arriba de sigma_min = 0.1, porque sigma nunca baja de ahi
        self.tol_mu = tol_mu          # en amperes
        self.mus = deque(maxlen=n)     # solo guarda los ultimos n valores. la ventana de mu y sigma para ver si convergio
        self.sigmas = deque(maxlen=n)

    def reiniciar(self):
        # despues de reabrir la exploracion empieza otra busqueda
        # la ventana vieja ya no sirve
        self.mus.clear()
        self.sigmas.clear()

    def convergio(self, mu, sigma):
        self.mus.append(mu)
        self.sigmas.append(sigma)
        if len(self.mus) < self.n:  # todavia no hay suficiente historia
            return False
        sigma_chica = max(self.sigmas) <= self.sigma_conv
        mu_quieta = (max(self.mus) - min(self.mus)) < self.tol_mu
        return sigma_chica and mu_quieta

def entrenar(env, politica, n_actualizaciones=300, n_muestras=10, alpha_mu=0.3, alpha_sigma=0.1, 
             detector=None, sigma_reapertura=1.5,
             criterio=None, parar_al_converger=False):
    """
    ciclo de entrenamiento

    en cada actualizacion: muestrear un batch, evaluarlo y actualizar la politica
    regresa el historial: una lista con un dict por actualizacion (para metricas y GIF)
    """
    historial = []
    evaluaciones = 0
    recompensa_acumulada = 0.0

    for paso in range(n_actualizaciones):
        # la politica antes de actualizar es la que genero las muestras de este batch
        mu, sigma = politica.mu, politica.sigma

        corrientes, recompensas, info = evaluar(env, politica, n_muestras)
        metricas = actualizar_politica(politica, corrientes, recompensas, alpha_mu, alpha_sigma)
        
        # si la potencia indica un cambio de caso, volver a explorar con sigma grande para encontrar el pico local del caso nuevo
        deteccion = detector.revisar(recompensas, sigma) if detector is not None else None
        if deteccion is not None:
            politica.reabrir_exploracion(sigma_reapertura)
        
        # criterio de paro: revisar si la politica ya convergio
        convergido = False
        if criterio is not None:
            if deteccion is not None:
                criterio.reiniciar()
            else:
                convergido = criterio.convergio(mu, sigma)

        evaluaciones += n_muestras
        recompensa_acumulada += recompensas.sum().item()

        historial.append({
            "paso": paso,
            "evaluaciones": evaluaciones,
            "recompensa_acumulada": recompensa_acumulada,
            "mu": mu,
            "sigma": sigma,
            "corrientes": corrientes.tolist(),
            "potencias": recompensas.tolist(),
            "deteccion": deteccion,  # none, cambio o sin_info
            "convergido": convergido,
            "casos": info["casos"],  # SOLO para graficar, el agente no lo usa !!!!!!!!!!!
            **metricas,              # grad_mu, grad_log_sigma, norma_grad, paso_mu, recompensa_media
        })

        if info["truncado"]:  # el entorno llego a max_steps
            break
        
        if convergido and parar_al_converger:  # criterio de paro cumplido wuuuuu
            break

    return historial

def historial_por_evaluacion(historial, I_max):
    """
    convierte el historial (1 registro por batch) al formato de actor_critic.py y gif.py
    1 registro por evaluacion, con las mismas claves que usa gen_gif

    REINFORCE no tiene critico: en 'V(critic) va la baseline (el promedio del batch),
    normalizada igual que en actor_critic: potencia / maxima potencia vista hasta ese momento.
    la ventaja tambien se calcula como en actor_critic: reward_norm - V
    """
    registros = []
    P_max_visto = 1e-8
    step = 0
    for h in historial:
        P_max_visto = max(P_max_visto, max(h["potencias"]))
        baseline = h["recompensa_media"] / P_max_visto

        for I, P, caso in zip(h["corrientes"], h["potencias"], h["casos"]):
            reward_norm = P / P_max_visto
            registros.append({
                # mismas de hector jejej
                "step": step,
                "caso": caso,                               # SOLO para graficar
                "I(Corriente)": min(max(I, 0.0), I_max),    # la corriente que aplico el entorno (recortada)
                "P(Potencia)": P,
                "reward_norm": reward_norm,
                "mu(media)": h["mu"],
                "sigma(desviacion)": h["sigma"],
                "V(critic)": baseline,                      # baseline de REINFORCE, no hay critico
                "advantage": reward_norm - baseline,        # que tanto supero la muestra a la baseline
                # extras que no usa el agente
                "paso": h["paso"],
                "grad_mu": h["grad_mu"],
                "norma_grad": h["norma_grad"],
                "deteccion": h["deteccion"] or "",
                "convergido": h["convergido"],
            })
            step += 1
    return registros


def guardar_historial_csv(registros, ruta):
    # igual que el otro pero para reinforce
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=registros[0].keys())
        escritor.writeheader()
        escritor.writerows(registros)
    print(f"historial guardado en: {ruta}")

if __name__ == "__main__":
    SEED = 0
    torch.manual_seed(SEED)

    env = Env(datos, steps_caso=500, max_steps=3000)
    env.reset(seed=SEED)

    politica = PoliticaGaussiana(I_max=env.I_max, sigma_inicial=1.0)

    detector = DetectorCambio(umbral=0.3, beta=0.3, sigma_vigilar=0.3)
    
    criterio = CriterioParo(n=10, sigma_conv=0.15, tol_mu=0.05)

    historial = entrenar(env, politica, n_actualizaciones=300, n_muestras=10,
                         alpha_mu=0.3, alpha_sigma=0.1,
                         detector=detector, sigma_reapertura=1.5, criterio=criterio)

    convergido_antes = False
    for h in historial:
        nueva_conv = h["convergido"] and not convergido_antes
        convergido_antes = h["convergido"]
        marca = h["deteccion"] or ("convergio" if nueva_conv else None)
        if h["paso"] % 10 != 0 and marca is None:
            continue  # imprimir 1 de cada 10 + donde detecta un cambio de caso o converge
        print(f"paso {h['paso']:3d} | eval {h['evaluaciones']:4d} | caso {h['casos'][-1]} | "
              f"mu {h['mu']:.2f} A | sigma {h['sigma']:.2f} | P media {h['recompensa_media']:6.1f} W | "
              f"|grad| {h['norma_grad']:.3f}" + (f"  <-- {marca}" if marca else ""))
    print(f"\nTotal: {historial[-1]['evaluaciones']} evaluaciones, "
          f"recompensa acumulada {historial[-1]['recompensa_acumulada']:.0f}")
    
    
    # carambolas
    registros = historial_por_evaluacion(historial, env.I_max)
    guardar_historial_csv(registros, CARPETA_RESULTADOS / "historial_reinforce.csv")

    from gif import gen_gif  #  lo tuve que mover acá porque
    # gif.py tiene from actor_critic import historial hasta arriba, y en actor_critic.py el if __name__ == "__main__": está comentado 
    # soo importar gen_gif corre el entrenamiento completo de actor-critic
    gen_gif(datos, historial=registros, salida=str(CARPETA_RESULTADOS), fpd=10, cada=10,
            muestras=10, nombre_salida="reinforce.gif")
    
    
    # con criterio de paro aplicado: entrenar en cada caso por separado hasta que converge
    print("\n")
    print("cada caso por separado hasta cumplir el criterio de paro:")
    for k in range(len(datos)):
        torch.manual_seed(SEED)
        env_k = Env([datos[k]], steps_caso=10**9, max_steps=10**6)
        env_k.reset(seed=SEED)
        politica_k = PoliticaGaussiana(I_max=env.I_max, sigma_inicial=1.0)  # mismo rango
        hist_k = entrenar(env_k, politica_k, n_actualizaciones=400, n_muestras=10,
                          detector=DetectorCambio(), criterio=CriterioParo(),
                          parar_al_converger=True)
        fin = hist_k[-1]
        estado = "convergio" if fin["convergido"] else "sorry... you're NOT a sigma"
        
        print(f"caso {k}: {estado} en {fin['evaluaciones']} evaluaciones | "
              f"mu {politica_k.mu:.2f} A | sigma {politica_k.sigma:.2f} | P media {fin['recompensa_media']:.1f} W")
    
    
"""
paso   0 | eval   10 | caso 3 | mu 4.45 A | sigma 1.00 | P media   63.2 W | |grad| 0.562
paso  10 | eval  110 | caso 3 | mu 4.28 A | sigma 0.78 | P media   61.3 W | |grad| 1.484
paso  20 | eval  210 | caso 3 | mu 4.40 A | sigma 0.57 | P media   63.9 W | |grad| 1.347
paso  30 | eval  310 | caso 3 | mu 4.96 A | sigma 0.41 | P media   68.6 W | |grad| 1.050
paso  40 | eval  410 | caso 3 | mu 4.88 A | sigma 0.24 | P media   69.3 W | |grad| 3.458
paso  50 | eval  510 | caso 1 | mu 5.05 A | sigma 0.16 | P media  166.6 W | |grad| 7.394
paso  60 | eval  610 | caso 1 | mu 5.48 A | sigma 0.13 | P media  174.3 W | |grad| 4.306
paso  70 | eval  710 | caso 1 | mu 5.54 A | sigma 0.10 | P media  175.4 W | |grad| 5.980
paso  80 | eval  810 | caso 1 | mu 5.57 A | sigma 0.10 | P media  169.1 W | |grad| 6.592
paso  90 | eval  910 | caso 1 | mu 5.52 A | sigma 0.10 | P media  176.0 W | |grad| 3.073
paso 100 | eval 1010 | caso 2 | mu 5.58 A | sigma 0.10 | P media    0.0 W | |grad| 0.000
paso 110 | eval 1110 | caso 2 | mu 5.58 A | sigma 0.10 | P media    0.0 W | |grad| 0.000
paso 120 | eval 1210 | caso 2 | mu 5.58 A | sigma 0.10 | P media    0.0 W | |grad| 0.000
paso 130 | eval 1310 | caso 2 | mu 5.58 A | sigma 0.10 | P media    0.0 W | |grad| 0.000
paso 140 | eval 1410 | caso 2 | mu 5.58 A | sigma 0.10 | P media    0.0 W | |grad| 0.000
paso 150 | eval 1510 | caso 0 | mu 5.57 A | sigma 0.10 | P media  182.8 W | |grad| 7.292
paso 160 | eval 1610 | caso 0 | mu 5.80 A | sigma 0.10 | P media  188.5 W | |grad| 4.205
paso 170 | eval 1710 | caso 0 | mu 6.04 A | sigma 0.10 | P media  193.0 W | |grad| 8.347
paso 180 | eval 1810 | caso 0 | mu 6.35 A | sigma 0.12 | P media  198.9 W | |grad| 4.368
paso 190 | eval 1910 | caso 0 | mu 6.61 A | sigma 0.10 | P media  200.9 W | |grad| 2.740
paso 200 | eval 2010 | caso 1 | mu 6.63 A | sigma 0.10 | P media  100.5 W | |grad| 4.541
paso 210 | eval 2110 | caso 1 | mu 6.93 A | sigma 0.12 | P media  103.4 W | |grad| 8.047
paso 220 | eval 2210 | caso 1 | mu 7.21 A | sigma 0.11 | P media  106.4 W | |grad| 7.810
paso 230 | eval 2310 | caso 1 | mu 7.51 A | sigma 0.10 | P media  109.4 W | |grad| 7.217
paso 240 | eval 2410 | caso 1 | mu 7.80 A | sigma 0.11 | P media  111.2 W | |grad| 8.412
paso 250 | eval 2510 | caso 0 | mu 8.02 A | sigma 0.12 | P media    0.0 W | |grad| 0.000
paso 260 | eval 2610 | caso 0 | mu 8.02 A | sigma 0.12 | P media    0.0 W | |grad| 0.000
paso 270 | eval 2710 | caso 0 | mu 8.02 A | sigma 0.12 | P media    0.0 W | |grad| 0.000
paso 280 | eval 2810 | caso 0 | mu 8.02 A | sigma 0.12 | P media    0.0 W | |grad| 0.000
paso 290 | eval 2910 | caso 0 | mu 8.02 A | sigma 0.12 | P media    0.0 W | |grad| 0.000

Total: 3000 evaluaciones, recompensa acumulada 270456

con la seed 0:
en el caso 3 mu todavía sigue subiendo, y llega a 69W que ya es acercado al maximo local de 75.2W
luego cambia al caso 1 y mu sigue subiendo hasta 176W, que es el maximo local de ese caso
luego caso 2 se queda atorado en 0W
pero en el caso 0 mu sigue subiendo hasta 200W, que sí es el maximo local de ese caso
luego caso 1 vuelve a subir hasta 111W, que es solo es el pico local
terminando en el caso 0 con 0W, igual atorado :v

parece que se atora cuando el cambio de caso lo deja en 0W o en otra joroba
"""

"""
paso   0 | eval   10 | caso 3 | mu 4.45 A | sigma 1.00 | P media   63.2 W | |grad| 0.562
paso  10 | eval  110 | caso 3 | mu 4.28 A | sigma 0.78 | P media   61.3 W | |grad| 1.484
paso  20 | eval  210 | caso 3 | mu 4.40 A | sigma 0.57 | P media   63.9 W | |grad| 1.347
paso  30 | eval  310 | caso 3 | mu 4.96 A | sigma 0.41 | P media   68.6 W | |grad| 1.050
paso  40 | eval  410 | caso 3 | mu 4.88 A | sigma 0.24 | P media   69.3 W | |grad| 3.458
paso  50 | eval  510 | caso 1 | mu 5.05 A | sigma 0.16 | P media  166.6 W | |grad| 7.394  <-- cambio
paso  60 | eval  610 | caso 1 | mu 5.07 A | sigma 0.94 | P media  144.0 W | |grad| 0.410
paso  70 | eval  710 | caso 1 | mu 4.81 A | sigma 0.60 | P media  158.4 W | |grad| 1.623
paso  80 | eval  810 | caso 1 | mu 5.14 A | sigma 0.41 | P media  152.8 W | |grad| 2.064
paso  90 | eval  910 | caso 1 | mu 5.18 A | sigma 0.26 | P media  171.6 W | |grad| 1.750
paso 100 | eval 1010 | caso 2 | mu 5.45 A | sigma 0.18 | P media    0.0 W | |grad| 0.000  <-- sin_info
paso 110 | eval 1110 | caso 2 | mu 2.74 A | sigma 1.69 | P media   23.3 W | |grad| 0.196
paso 120 | eval 1210 | caso 2 | mu 1.92 A | sigma 1.22 | P media   20.7 W | |grad| 0.873
paso 130 | eval 1310 | caso 2 | mu 1.90 A | sigma 0.82 | P media   30.7 W | |grad| 0.807
paso 140 | eval 1410 | caso 2 | mu 1.92 A | sigma 0.66 | P media   34.5 W | |grad| 0.696
paso 150 | eval 1510 | caso 0 | mu 2.15 A | sigma 0.61 | P media   80.8 W | |grad| 1.223
paso 160 | eval 1610 | caso 0 | mu 3.56 A | sigma 0.62 | P media  129.2 W | |grad| 0.701
paso 170 | eval 1710 | caso 0 | mu 4.98 A | sigma 0.58 | P media  165.7 W | |grad| 1.469
paso 180 | eval 1810 | caso 0 | mu 6.10 A | sigma 0.44 | P media  196.3 W | |grad| 1.140
paso 190 | eval 1910 | caso 0 | mu 6.46 A | sigma 0.31 | P media  200.1 W | |grad| 0.155
paso 200 | eval 2010 | caso 1 | mu 6.54 A | sigma 0.20 | P media   99.6 W | |grad| 2.257  <-- cambio
paso 210 | eval 2110 | caso 1 | mu 4.88 A | sigma 1.01 | P media  130.6 W | |grad| 0.708
paso 220 | eval 2210 | caso 1 | mu 5.08 A | sigma 0.72 | P media  155.7 W | |grad| 1.032
paso 230 | eval 2310 | caso 1 | mu 5.18 A | sigma 0.42 | P media  161.3 W | |grad| 1.545
paso 240 | eval 2410 | caso 1 | mu 5.43 A | sigma 0.27 | P media  156.2 W | |grad| 2.849
paso 250 | eval 2510 | caso 0 | mu 5.29 A | sigma 0.18 | P media  177.4 W | |grad| 7.427
paso 260 | eval 2610 | caso 0 | mu 5.93 A | sigma 0.24 | P media  192.0 W | |grad| 4.813
paso 270 | eval 2710 | caso 0 | mu 6.42 A | sigma 0.25 | P media  199.0 W | |grad| 3.367
paso 280 | eval 2810 | caso 0 | mu 6.57 A | sigma 0.15 | P media  200.3 W | |grad| 4.751
paso 290 | eval 2910 | caso 0 | mu 6.65 A | sigma 0.10 | P media  200.7 W | |grad| 1.692

Total: 3000 evaluaciones, recompensa acumulada 377705

igual con la seed 0 para comparar, pero ahora con el detector de cambio de caso
en el caso 3 mu todavía sigue subiendo, y llega a 69W ok sigue igual que antes
luego cambia al caso 1 y mu sigue subiendo hasta 171.6W, es distinto a la version sin detector que se quedo en 176W,
cuando pasa al caso 2 se queda atorado en 0W, pero el detector detecta que la potencia media bajo mucho y reabre la exploracion con sigma grande, 
entonces mu baja a 1.9A y encuentra un pico local de 34.5W
luego pasa al caso 0 y mu sube hasta 200.7W, que sí es el maximo local de ese caso por fin jala !!!!!
luego pasa al caso 1 y mu sube hasta 156.2W, al menos no se queda atorado en 111W como antes aunque no llega al maximo local de 176W
al final termina en el caso 0 con 200.7W, que sí es el maximo local de ese caso entonces todo ok

yo digo que así es good enough, porque ya no se atora en picos locales, especialmente en el caso 2 que era un dolor de huevos

lo que sí es que si un cambio de caso casi no altera la potencia en el punto donde está mu no se detecta por el umbral, 
igual y es cuestion de jugar con el umbral y minmaxear los valores, pero no hay tiempo chicos jaja lol

no recuerdo bien la seed pero en una me pasó que del caso 2 al caso 3 con mu de como 1.5A la potencia pasa de 46.7W a 50.6W
y el agente se queda en el pico de 50.6W del caso 3 aunque existe uno de 75.2W

y pues lolazo, supongo que peor es nada porque al menos ya no se queda atorado en 0W como antes
yo creo que si se le da chance de explorar con sigma grande puede encontrar el pico local del caso nuevo, aunque no sea el global

igual y se puede hacer un detector de cambio de caso más inteligente, pero no hay tiempo xd
con solo la potencia observada no hay forma de saber que existe un pico mas alto sin salir a explorar
puede ser si se le da la info al agente pero aparte de que es trampa supongo que tampoco es realista para el problema irl
"""

"""
paso   0 | eval   10 | caso 3 | mu 4.45 A | sigma 1.00 | P media   63.2 W | |grad| 0.562
paso  10 | eval  110 | caso 3 | mu 4.28 A | sigma 0.78 | P media   61.3 W | |grad| 1.484
paso  20 | eval  210 | caso 3 | mu 4.40 A | sigma 0.57 | P media   63.9 W | |grad| 1.347
paso  30 | eval  310 | caso 3 | mu 4.96 A | sigma 0.41 | P media   68.6 W | |grad| 1.050
paso  40 | eval  410 | caso 3 | mu 4.88 A | sigma 0.24 | P media   69.3 W | |grad| 3.458
paso  50 | eval  510 | caso 1 | mu 5.05 A | sigma 0.16 | P media  166.6 W | |grad| 7.394  <-- cambio
paso  60 | eval  610 | caso 1 | mu 5.07 A | sigma 0.94 | P media  144.0 W | |grad| 0.410
paso  70 | eval  710 | caso 1 | mu 4.81 A | sigma 0.60 | P media  158.4 W | |grad| 1.623
paso  80 | eval  810 | caso 1 | mu 5.14 A | sigma 0.41 | P media  152.8 W | |grad| 2.064
paso  90 | eval  910 | caso 1 | mu 5.18 A | sigma 0.26 | P media  171.6 W | |grad| 1.750
paso 100 | eval 1010 | caso 2 | mu 5.45 A | sigma 0.18 | P media    0.0 W | |grad| 0.000  <-- sin_info
paso 110 | eval 1110 | caso 2 | mu 2.74 A | sigma 1.69 | P media   23.3 W | |grad| 0.196
paso 120 | eval 1210 | caso 2 | mu 1.92 A | sigma 1.22 | P media   20.7 W | |grad| 0.873
paso 130 | eval 1310 | caso 2 | mu 1.90 A | sigma 0.82 | P media   30.7 W | |grad| 0.807
paso 140 | eval 1410 | caso 2 | mu 1.92 A | sigma 0.66 | P media   34.5 W | |grad| 0.696
paso 150 | eval 1510 | caso 0 | mu 2.15 A | sigma 0.61 | P media   80.8 W | |grad| 1.223
paso 160 | eval 1610 | caso 0 | mu 3.56 A | sigma 0.62 | P media  129.2 W | |grad| 0.701
paso 170 | eval 1710 | caso 0 | mu 4.98 A | sigma 0.58 | P media  165.7 W | |grad| 1.469
paso 180 | eval 1810 | caso 0 | mu 6.10 A | sigma 0.44 | P media  196.3 W | |grad| 1.140
paso 190 | eval 1910 | caso 0 | mu 6.46 A | sigma 0.31 | P media  200.1 W | |grad| 0.155
paso 200 | eval 2010 | caso 1 | mu 6.54 A | sigma 0.20 | P media   99.6 W | |grad| 2.257  <-- cambio
paso 210 | eval 2110 | caso 1 | mu 4.88 A | sigma 1.01 | P media  130.6 W | |grad| 0.708
paso 220 | eval 2210 | caso 1 | mu 5.08 A | sigma 0.72 | P media  155.7 W | |grad| 1.032
paso 230 | eval 2310 | caso 1 | mu 5.18 A | sigma 0.42 | P media  161.3 W | |grad| 1.545
paso 240 | eval 2410 | caso 1 | mu 5.43 A | sigma 0.27 | P media  156.2 W | |grad| 2.849
paso 250 | eval 2510 | caso 0 | mu 5.29 A | sigma 0.18 | P media  177.4 W | |grad| 7.427
paso 260 | eval 2610 | caso 0 | mu 5.93 A | sigma 0.24 | P media  192.0 W | |grad| 4.813
paso 270 | eval 2710 | caso 0 | mu 6.42 A | sigma 0.25 | P media  199.0 W | |grad| 3.367
paso 280 | eval 2810 | caso 0 | mu 6.57 A | sigma 0.15 | P media  200.3 W | |grad| 4.751
paso 290 | eval 2910 | caso 0 | mu 6.65 A | sigma 0.10 | P media  200.7 W | |grad| 1.692
paso 296 | eval 2970 | caso 0 | mu 6.66 A | sigma 0.10 | P media  200.6 W | |grad| 1.020  <-- convergio

Total: 3000 evaluaciones, recompensa acumulada 377705


cada caso por separado hasta cumplir el criterio de paro:
caso 0: convergio en 560 evaluaciones | mu 6.62 A | sigma 0.10 | P media 200.7 W
caso 1: convergio en 780 evaluaciones | mu 5.57 A | sigma 0.10 | P media 175.2 W
caso 2: convergio en 1270 evaluaciones | mu 2.97 A | sigma 0.10 | P media 40.7 W
caso 3: convergio en 610 evaluaciones | mu 5.15 A | sigma 0.10 | P media 74.7 W
"""