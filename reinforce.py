import torch
import torch.nn as nn


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
            mu_inicial = self.I_max / 2  # empezar a la mitad del rango de corriente

        # logit es la inversa de la sigmoide: así mu arranca exactamente en mu_inicial
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