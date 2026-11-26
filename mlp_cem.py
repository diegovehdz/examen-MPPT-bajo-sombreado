import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from environment import Env, datos

P_ESCALA = 100.0  # solo escala la entrada de la red, NO es el maximo de la curva


class PoliticaMLP(nn.Module):
    """
    entra [P_anterior, I_anterior] y sale la corriente I
    los pesos no se entrenan con gradiente: los elige CEM
    """
    def __init__(self, I_max, ocultas=8):
        super().__init__()
        self.I_max = float(I_max)
        self.red = nn.Sequential(
            nn.Linear(2, ocultas),
            nn.Tanh(),
            nn.Linear(ocultas, 1),
            nn.Sigmoid(),  # sale en [0, 1], luego * I_max
        )

    @torch.no_grad()  # no hay backpropation en el CEM
    def forward(self, P_ant, I_ant):
        x = torch.tensor([P_ant / P_ESCALA, I_ant / self.I_max], dtype=torch.float32)
        return self.red(x).item() * self.I_max

    def n_parametros(self):
        return parameters_to_vector(self.parameters()).numel()

    def cargar_pesos(self, theta):
        vector_to_parameters(theta, self.parameters())
        
def evaluar(env, red, theta, P_ini, I_ini, n_pasos=3):
    """
    carga theta en la red y la deja controlar el panel n_pasos seguidos
    la red ve lo que paso en el paso anterior [P_ant, I_ant] -> I

    cada paso es 1 llamada al step = 1 evaluacion de la curva como en todo lo demás equisde
    score = potencia media de los n_pasos (con esto CEM ordena y elige la elite)

    regresa:
    - score
    - corrientes, potencias: para el historial y el GIF
    - estado_final: (P, I) del ultimo paso, para arrancar la siguiente iteracion desde ahi
    - info_lote: SOLO para visualizar, el agente no lo usa !!!
    """
    red.cargar_pesos(theta)
    P_ant, I_ant = P_ini, I_ini
    corrientes, potencias, casos, truncado = [], [], [], False

    for _ in range(n_pasos):
        I = red(P_ant, I_ant)
        _, P, _, truncated, info = env.step(np.array([I], dtype=np.float32))
        corrientes.append(I)
        potencias.append(P)
        casos.append(info["caso"])
        truncado = truncado or truncated
        P_ant, I_ant = P, I

    score = float(np.mean(potencias))
    return score, corrientes, potencias, (P_ant, I_ant), {"casos": casos, "truncado": truncado}


def entrenar(env, red, P_inicial, I_inicial=4.2, n_iteraciones=200, n_candidatos=20,
             frac_elite=0.2, n_pasos=3, alpha=0.5, sigma_inicial=1.0, sigma_min=0.05):
    """
    CEM sobre los pesos de la red: es el mismo bucle que el CEM clasico,
    pero lo que se muestrea es el vector theta con todos los pesos, no una corriente
    -> mu y sigma son vectores con una entrada por peso (gaussiana diagonal)

    en cada iteracion:
    1. muestrear n_candidatos redes: theta ~ N(mu, sigma^2)
    2. evaluar cada una con evaluar(), todas desde el mismo estado (P, I) para que sea justo
    3. elite: las frac_elite redes con mayor potencia media
    4. acercar mu y sigma a la media y desviacion de la elite (suavizado con alpha como el clasico)
       sigma no baja de sigma_min para que no deje de explorar del todo

    regresa el historial: una lista con un dict por iteracion (para metricas y GIF)
    """
    d = red.n_parametros()
    mu = torch.zeros(d)                               # red "media": todos los pesos en 0 -> I = I_max/2
    sigma = torch.full((d,), sigma_inicial)
    n_elite = max(2, int(n_candidatos * frac_elite))  # minimo 2 para poder sacar la desviacion

    P_ant, I_ant = P_inicial, I_inicial
    historial = []
    evaluaciones = 0
    recompensa_acumulada = 0.0

    for paso in range(n_iteraciones):
        # 1. muestrear: cada renglon es un vector de pesos completo = una red candidata
        thetas = mu + sigma * torch.randn(n_candidatos, d)

        # 2. evaluar todas las redes desde el mismo estado
        resultados = [evaluar(env, red, theta, P_ant, I_ant, n_pasos) for theta in thetas]
        scores = torch.tensor([r[0] for r in resultados], dtype=torch.float32)

        # 3. elite
        scores_elite, idx_elite = torch.topk(scores, k=n_elite)
        thetas_elite = thetas[idx_elite]

        # 4. actualizar la distribucion de pesos
        mu_anterior = mu
        deteccion = None
        if scores_elite.max().item() <= 0.0:
            # todas dieron0 W (la corriente quedo pasando la I_sc del caso nuevo)
            # no hay de donde aprender: reabrir la exploracion como el clasico, pero sin perder mu
            sigma = torch.full((d,), sigma_inicial)
            deteccion = "sin_info"
        else:
            mu = alpha * thetas_elite.mean(dim=0) + (1 - alpha) * mu
            sigma = torch.clamp(alpha * thetas_elite.std(dim=0) + (1 - alpha) * sigma, min=sigma_min)

        # la siguiente iteracion arranca desde donde termino la mejor red
        P_ant, I_ant = resultados[idx_elite[0].item()][3]

        # salida de la red media (theta = mu) desde ese estado
        # es "la politica" para el GIF
        red.cargar_pesos(mu)
        I_red_media = red(P_ant, I_ant)

        corrientes = [I for r in resultados for I in r[1]]
        potencias = [P for r in resultados for P in r[2]]
        casos = [c for r in resultados for c in r[4]["casos"]]
        evaluaciones += len(potencias)
        recompensa_acumulada += sum(potencias)

        historial.append({
            "paso": paso,
            "evaluaciones": evaluaciones,
            "recompensa_acumulada": recompensa_acumulada,
            "corrientes": corrientes,
            "potencias": potencias,
            "casos": casos,                                     # SOLO para graficar, el agente no lo usa !!!
            "score_medio": scores.mean().item(),
            "score_elite": scores_elite.mean().item(),
            "I_red_media": I_red_media,
            "sigma_media": sigma.mean().item(),
            "norma_paso_mu": (mu - mu_anterior).norm().item(),  # CEM no tiene gradiente, esto es lo mas parecido
            "deteccion": deteccion,
        })

        if any(r[4]["truncado"] for r in resultados):  # el entorno llego a max_steps
            break

    return historial


if __name__ == "__main__":
    SEED = 0
    torch.manual_seed(SEED)

    env = Env(datos, steps_caso=500, max_steps=3000)  # igual que reinforce para poder comparar con ese
    obs, info = env.reset(seed=SEED)

    red = PoliticaMLP(env.I_max)
    historial = entrenar(env, red, P_inicial=float(obs[0]))

    for h in historial:
        if h["paso"] % 5 != 0 and h["deteccion"] is None:
            continue  # imprimir 1 de cada 5 + donde reabre la exploracion, 1 de cada 10 ya no jala acá :'v
        print(f"paso {h['paso']:3d} | eval {h['evaluaciones']:4d} | caso {h['casos'][-1]} | "
              f"I red {h['I_red_media']:.2f} A | sigma {h['sigma_media']:.3f} | "
              f"P media {h['score_medio']:6.1f} W | P elite {h['score_elite']:6.1f} W | "
              f"|paso mu| {h['norma_paso_mu']:.3f}" + (f"  <-- {h['deteccion']}" if h["deteccion"] else ""))
    print(f"\nTotal: {historial[-1]['evaluaciones']} evaluaciones, "
          f"recompensa acumulada {historial[-1]['recompensa_acumulada']:.0f}")