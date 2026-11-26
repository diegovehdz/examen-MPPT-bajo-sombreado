import math
import numpy as np
import torch
from environment import Env, datos


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


def entrenar(env, politica, n_actualizaciones=300, n_muestras=10, alpha_mu=0.3, alpha_sigma=0.1):
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
            "casos": info["casos"],  # SOLO para graficar, el agente no lo usa !!!!!!!!!!!
            **metricas,              # grad_mu, grad_log_sigma, norma_grad, paso_mu, recompensa_media
        })

        if info["truncado"]:  # el entorno llego a max_steps
            break

    return historial


if __name__ == "__main__":
    SEED = 67
    torch.manual_seed(SEED)

    env = Env(datos, steps_caso=500, max_steps=3000)
    env.reset(seed=SEED)

    politica = PoliticaGaussiana(I_max=env.I_max, sigma_inicial=1.0)

    historial = entrenar(env, politica, n_actualizaciones=300, n_muestras=10,
                         alpha_mu=0.3, alpha_sigma=0.1)

    for h in historial[::10]:  # imprimir 1 de cada 10 actualizaciones porque son un chingo de steps
        print(f"paso {h['paso']:3d} | eval {h['evaluaciones']:4d} | caso {h['casos'][-1]} | "
              f"mu {h['mu']:.2f} A | sigma {h['sigma']:.2f} | P media {h['recompensa_media']:6.1f} W | "
              f"|grad| {h['norma_grad']:.3f}")
    print(f"\nTotal: {historial[-1]['evaluaciones']} evaluaciones, "
          f"recompensa acumulada {historial[-1]['recompensa_acumulada']:.0f}")
    
    
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
"""