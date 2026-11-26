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