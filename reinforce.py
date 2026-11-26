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