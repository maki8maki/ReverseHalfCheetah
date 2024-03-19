import gymnasium as gym

from wrapper import ImageObservationWrapper, RepeatActionWrapper

def make_env(env_name: str, config: dict = {}):
    env = ImageObservationWrapper(gym.make(env_name, render_mode='rgb_array', **config))
    env = RepeatActionWrapper(env, skip=2)
    return env
