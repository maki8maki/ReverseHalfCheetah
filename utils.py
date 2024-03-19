import gymnasium as gym
import numpy as np
import random
import torch as th

from wrapper import ImageObservationWrapper, RepeatActionWrapper

def make_env(env_name: str, config: dict = {}):
    env = ImageObservationWrapper(gym.make(env_name, render_mode='rgb_array', **config))
    env = RepeatActionWrapper(env, skip=2)
    return env

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
