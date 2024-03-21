import numpy as np
import torch as th

def normalize_observation(observation: np.ndarray, min=0.0, max=255.0):
    obs = observation.astype(np.float32)
    normalized_obs = (obs - min) / (max - min) - 0.5
    return normalized_obs

def lambda_target(rewards: th.Tensor, values: th.Tensor, gamma, lambda_):
    V_lambda = th.zeros_like(rewards, device=rewards.device)

    H = rewards.shape[0] - 1
    V_n = th.zeros_like(rewards, device=rewards.device)
    V_n[H] = values[H]
    for n in range(1, H+1):
        V_n[:-n] = (gamma ** n) * values[n:]
        for k in range(1, n+1):
            if k == n:
                V_n[:-n] += (gamma ** (n-1)) * rewards[k:]
            else:
                V_n[:-n] += (gamma ** (k-1)) * rewards[k:-n+k]

        if n == H:
            V_lambda += (lambda_ ** (H-1)) * V_n
        else:
            V_lambda += (1 - lambda_) * (lambda_ ** (n-1)) * V_n

    return V_lambda
