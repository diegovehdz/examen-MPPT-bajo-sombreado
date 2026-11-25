import numpy as np
import torch 
from torch.distributions import Normal

datos0 = np.load("datasets/caso_0.npy")
datos1 = np.load("datasets/caso_1.npy")
datos2 = np.load("datasets/caso_2.npy")
datos3 = np.load("datasets/caso_3.npy")

torch.manual_seed(42)
np.random_seed(42)

def log_normal(distribucion, media, var):
    return -0.5 * ((distribucion - media) / var) ** 2 - torch.log(var) - 0.5 * np.log(2 * np.pi)

def ppo(media_0, var_0, v_0, I_min, I_max, f, N=10, step=300, K=4, epsilon=0.2,
        alpha_media=0.01, alpha_var=0.01, alpha_v=0.1,
        var_min=0.05, var_max=2.0):
        # Paso 1: Inicialización de parámetros de política y valores iniciales
        media = torch.tensor(float(media_0), requires_grad=True)
        var = torch.tensor(float(var_0), requires_grad=True)
        v = torch.tensor(float(v_0), requires_grad=True)
        escala_P = 1e-8
        historial = []

        # Paso 2: For loop
        for i in range (step):
            with torch.no_grad():
                # Paso 3: recolectar el lote trayectorias con la política vieja
                media_old, var_old = media.clone(), var.clone()
                distribucion = media_old + var_old * torch.randn(N)
                i = distribucion.clamp(I_min, I_max)
                p = torch.tensor([f(float(x)) for x in i])
                log_p_viejo = log_normal(distribucion, media_old, var_old)


                # step 4: se estandariza reward
                for j in range(N):
                    if p[j].item() > escala_P:
                        escala_P = p[j].item()

                reward = []
                for j in range(N):
                    r = p[j].item() / escala_P
                    reward.append(r)
                reward = torch.tensor(reward)
                # paso 5: Se saca el advantage con el critic actual
                advantage = reward - v



