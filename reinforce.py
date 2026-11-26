import torch
import torch.nn as nn
import numpy as np

class PoliticaGaussiana(nn.Module):
    """
    politica pi(I) = N(mu, sigma)

    - mu se guarda como theta_mu y pasa por una sigmoide:
      mu = I_max * sigmoid(theta_mu)
      mu siempre queda dentro de [0, I_max]
      
    - sigma es log_sigma y se recupera con exp()
      sigma siempre es positiva
    """

    def __init__(self, I_max, mu_inicial=None, sigma_inicial=1.0, sigma_min=0.1):
        super().__init__()
        self.I_max = float(I_max)
        self.sigma_min = sigma_min
        if mu_inicial is None:
            mu_inicial = self.I_max / 2

        # logit es la inversa de la sigmoide para que mu arranque  en mu_inicial
        self.theta_mu = nn.Parameter(torch.logit(torch.tensor(mu_inicial / self.I_max)))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma_inicial))))

    def forward(self):
        mu = self.I_max * torch.sigmoid(self.theta_mu)
        sigma = self.log_sigma.exp().clamp(min=self.sigma_min)
        return mu, sigma

    def distribucion(self):
        # la gaussiana N(mu, sigma) de PyTorch sabe muestrear y calcular log_prob
        mu, sigma = self()
        return torch.distributions.Normal(mu, sigma)
    
def evaluar(env, politica, n_muestras=10):
    """
    muestrea n corrientes de la política y evalúa cada una en el entorno.

    cada muestra es 1 llamada al step, 
    un step es una evaluación de la curva
    
    regresa:
    - corrientes:  tensor con las corrientes que salieron de la política
    - recompensas: tensor con la W que regresó el entorno 
    - info_lote:   SOLO para visualizar /// el agente no debe usarlo !!!!!!!!
    """
    # el gradiente sale de log_prob en la pieza 3
    corrientes = politica.distribucion().sample((n_muestras,))

    recompensas, casos, truncado = [], [], False
    for I in corrientes:
        # el entorno espera un arreglo de forma (1,)
        # recorta I a [0, I_max]
        _, r, _, truncated, info = env.step(np.array([I.item()], dtype=np.float32))
        recompensas.append(r)
        casos.append(info["caso"])
        truncado = truncado or truncated

    recompensas = torch.tensor(recompensas, dtype=torch.float32)
    return corrientes, recompensas, {"casos": casos, "truncado": truncado}

def actualizar_politica(politica, optimizador, corrientes, recompensas):
    """
    un paso de REINFORCE con baseline (el promedio del batch)

    1. ventaja:  A_i = (r_i - promedio(r)) / desviación(r)
    2. pérdida:  -promedio(A_i * log pi(I_i))           el signo menos es porque pytorch minimiza y debemos maximizar la potencia esperada 
    3. gradiente y paso del optimizador

    regresa un dict con valores para métricas y para el GIF
    """
    # ventaja: qué tan buena fue cada muestra comparada con las demás del mismo batch
    # (se truncaba pero poner +1e-8 evita dividir entre 0 si todas las recompensas son iguales ????)
    ventajas = (recompensas - recompensas.mean()) / (recompensas.std() + 1e-8)

    # 1. pérdida
    log_probs = politica.distribucion().log_prob(corrientes)
    perdida = -(ventajas * log_probs).mean()

    # calcular gradientes y mover los parametros
    optimizador.zero_grad()     # borrar los gradientes del lote anterior
    perdida.backward()          # calcular d(perdida)/d(theta_mu) y d(perdida)/d(log_sigma)

    # 3. gradiente y paso del optimizador
    # positivo = el parametro va a subir
    grad_mu = -politica.theta_mu.grad.item()
    grad_log_sigma = -politica.log_sigma.grad.item()
    norma_grad = (grad_mu ** 2 + grad_log_sigma ** 2) ** 0.5

    optimizador.step() # aplicar la actualización

    return {
        "perdida": perdida.item(),
        "grad_mu": grad_mu,
        "grad_log_sigma": grad_log_sigma,
        "norma_grad": norma_grad,
        "recompensa_media": recompensas.mean().item(),
    }
    
# torch.optim.Adam(politica.parameters(), lr=0.1)

"""
.venv\Scripts\python.exe -c "import torch; torch.manual_seed(0); from environment import Env, datos; from reinforce import PoliticaGaussiana, evaluar, actualizar_politica; env = Env(datos, steps_caso=500, max_steps=10000); env.reset(seed=0); p = PoliticaGaussiana(I_max=env.I_max); opt = torch.optim.Adam(p.parameters(), lr=0.1); print('antes:', [round(x.item(), 3) for x in p()]); I, r, _ = evaluar(env, p); print(actualizar_politica(p, opt, I, r)); print('despues:', [round(x.item(), 3) for x in p()])"

OUT:
antes: [4.451, 1.0]
{'perdida': -0.27824658155441284, 'grad_mu': -0.18126007914543152, 'grad_log_sigma': -0.5564925074577332, 'norma_grad': 0.5852684231601798, 'recompensa_media': 63.17902374267578}
despues: [4.229, 0.905]

mu se movio muy poquito a la izquierda
"""

"""
.venv\Scripts\python.exe -c "import torch; torch.manual_seed(0); from environment import Env, datos; from reinforce import PoliticaGaussiana, evaluar, actualizar_politica; env = Env(datos, steps_caso=500, max_steps=10000); env.reset(seed=0); p = PoliticaGaussiana(I_max=env.I_max); opt = torch.optim.Adam(p.parameters(), lr=0.1); h = [actualizar_politica(p, opt, *evaluar(env, p)[:2]) for _ in range(30)]; print('mu, sigma:', [round(x.item(), 3) for x in p()], '| R media:', round(h[0]['recompensa_media'], 1), '->', round(h[-1]['recompensa_media'], 1))"

OUT:
mu, sigma: [5.182, 0.1] | R media: 63.2 -> 74.7

ya aprende!!! 

en el caso 3 mu llego a 5.18A 
si el pico está en 5.19A con 75.15 W entonces está casi empatado con el global
"""