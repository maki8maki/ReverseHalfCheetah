import gymnasium as gym
from matplotlib import animation
import matplotlib.pyplot as plt
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

def anim(frames, titles=None, filename=None, show=True):
    plt.figure(figsize=(frames[0].shape[1], frames[0].shape[0]), dpi=144)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
        if titles is not None:
            plt.title(titles[i], fontsize=32)

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    if filename is not None:
        anim.save(filename, writer="ffmpeg")
    if show:
        plt.show()
