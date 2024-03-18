from typing import Any, Dict, SupportsFloat, Tuple
import gymnasium as gym

class ImageObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        obs, _ = self.env.reset()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs.shape, dtype='uint8')
    
    def observation(self, observation: Any) -> Any:
        return self.env.render()

class RepeatActionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
    
    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, truncated, terminated, info
