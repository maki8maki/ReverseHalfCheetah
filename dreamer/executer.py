import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dreamer.buffer import ReplayBuffer
from dreamer.core import Agent, Dreamer
from utils import make_env

class Executer():
    def __init__(self, env_name: str, env_config: dict={}, dreamer_config: dict={}, buffer_capacity=200000, log_dir='logs', vervose=False):
        self.env = make_env(env_name, env_config)
        self.dreamer = Dreamer(self.env.action_space.shape[0], **dreamer_config)
        self.buffer = ReplayBuffer(buffer_capacity, self.env.observation_space.shape, self.env.action_space.shape[0])

        self.log_dir = log_dir
        self.verbose = vervose
        self.make_aliases()
    
    def make_aliases(self):
        self.encoder = self.dreamer.encoder
        self.rssm = self.dreamer.rssm
        self.obs_model = self.dreamer.obs_model
        self.reward_model = self.dreamer.reward_model
        self.value_model = self.dreamer.value_model
        self.action_model = self.dreamer.action_model
    
    def learn(self, start_episodes=5, all_episodes=300, eval_interval=10, model_save_interval=20, collect_interval=100, action_noise_var=0.3,
              batch_size=50, chunk_length=50, free_nats=3, clip_grad_norm=100, imagination_horizon=15, gamma=0.9, lambda_=0.95):
        writer = SummaryWriter(self.log_dir)
        for episode in range(start_episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.buffer.push(obs, action, reward, done)
                obs = next_obs
        for episode in tqdm(range(start_episodes, all_episodes)):
            policy = Agent(self.encoder, self.rssm, self.action_model)

            obs, _ = self.env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = policy(obs)
                action += np.random.normal(0, np.sqrt(action_noise_var), self.env.action_space.shape[0])
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.buffer.push(obs, action, reward, done)
                obs = next_obs
                total_reward += reward
            
            writer.add_scalar('train/total_reward', total_reward, episode)
            if (self.verbose):
                tqdm.write('episode [%4d/%4d] is collected. Total reward is %f' % (episode+1, all_episodes, total_reward))

            for update_step in range(collect_interval):
                observations, actions, rewards, _ = self.buffer.sample(batch_size, chunk_length)

                losses = self.dreamer.update(observations, actions, rewards, free_nats, clip_grad_norm, imagination_horizon, gamma, lambda_)

                log = 'update_step: %3d' % (update_step+1)
                total_update_step = episode * collect_interval + update_step
                for key, value in losses.items():
                    log += f', {key}: ' + '%.5f' % value.item()
                    writer.add_scalar(f'train/{key}', value.item(), total_update_step)
                
                if (self.verbose):
                    tqdm.write(log)

            if (episode+1) % eval_interval == 0:
                total_reward = self.evaluate()
                writer.add_scalar('test/total_reward', total_reward, episode+1)
                tqdm.write('Total test reward at episode [%4d/%4d] is %f' % (episode+1, all_episodes, total_reward))
            
            if (episode+1) % model_save_interval == 0:
                self.save(self.log_dir, 'episode_%04d.pth' % (episode+1))
        writer.close()
    
    def evaluate(self):
        policy = Agent(self.encoder, self.rssm, self.action_model)
        obs, _ = self.env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = policy(obs, training=False)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            obs = next_obs
            total_reward += reward
        return total_reward
    
    def save(self, log_dir='logs', name='params.pth'):
        os.makedirs(log_dir, exist_ok=True)
        self.dreamer.save(os.path.join(log_dir, name))
    
    def close(self):
        self.env.close()
