import numpy as np
import torch 
from torch.distributions import Normal
from environment import Env, datos

datos0 = np.load("datasets/caso_0.npy")
datos1 = np.load("datasets/caso_1.npy")
datos2 = np.load("datasets/caso_2.npy")
datos3 = np.load("datasets/caso_3.npy")

torch.manual_seed(42)
np.random.seed(42)

def log_normal(distribucion, media, var):
    return -0.5 * ((distribucion - media) / var) ** 2 - torch.log(var) - 0.5 * np.log(2 * np.pi)

def ppo(media_0, var_0, v_0, I_min, I_max, env, N=10, step=300, K=4, epsilon=0.2,
        alpha_media=0.01, alpha_var=0.01, alpha_v=0.1,
        var_min=0.05, var_max=2.0):
        # Paso 1: Inicialización de parámetros de política y valores iniciales
        media = torch.tensor(float(media_0), requires_grad=True)
        var = torch.tensor(float(var_0), requires_grad=True)
        v = torch.tensor(float(v_0), requires_grad=True)
        escala_P = 1e-8
        historial = []

        # Paso 2: For loop
        for y in range (step):
            with torch.no_grad():
                # Paso 3: recolectar el lote trayectorias con la política vieja
                media_old, var_old = media.clone(), var.clone()
                distribucion = media_old + var_old * torch.randn(N)
                i = distribucion.clamp(I_min, I_max)
                p = []
                for x in i:
                    obs, reward, terminated, truncated, info = env.step(np.array([float(x)]))
                    p.append(reward)
                p = torch.tensor(p)
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

                # Paso 6: Se actulaiza el actor, se crea la función clipping (PPO)
                for z in range (K):
                    ratio = torch.exp(log_normal(distribucion, media, var) - log_p_viejo)
                    clipping = ((advantage >= 0) & (ratio > 1 + epsilon)) | ((advantage < 0) & (ratio < 1 - epsilon))
                    peso = torch.where(clipping, torch.zeros_like(ratio), ratio * advantage)

                    gradiente_media = (distribucion - media) / var**2
                    gradiente_var = ((distribucion - media)**2 / var**3) - (1 / var)

                    gradiente_media_ppo = (peso * gradiente_media).mean()
                    gradiente_var_ppo = (peso * gradiente_var).mean()

                    media = (media + alpha_media * gradiente_media_ppo).clamp(I_min, I_max)
                    var = (var + alpha_var * gradiente_var_ppo).clamp(var_min, var_max)
                

                
                # paso 7: Se actualiza el critic
                v = v + alpha_v * (reward.mean() - v)

                historial.append(dict(paso=i, media=media.item(), var=var.item(), v=v.item(),
                                    reward=reward.mean().item(),
                                    gradiente_media=gradiente_media_ppo.item(),
                                    gradiente_var=gradiente_var_ppo.item(),
                                    i=i.numpy().copy(), p=p.numpy().copy()))

        return media.item(), historial

if __name__ == "__main__":
    SEED = 42
    env = Env(datos, steps_caso=50, max_steps=1000)
    obs, info = env.reset(seed=SEED)

    media_final, historial = ppo(
        media_0=3.8,
        var_0=0.2,
        v_0=0.0,
        I_min=env.I_min,
        I_max=env.I_max,
        env=env,
        N=10,
        step=300,
        K=4,
    )
    print(media_final, historial[-1])



