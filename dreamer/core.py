from typing import Dict
import numpy as np
import torch as th
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from dreamer.utils import normalize_observation, lambda_target

class Encoder(nn.Module):
    def __init__(self, channel=3, activation=nn.ReLU(inplace=True)):
        super().__init__()
        channels = [channel, 32, 64, 128, 256]
        kernel_size = 4
        stride = 2

        modules = []
        for i in range(len(channels)-1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernel_size, stride=stride),
                    activation,
                )
            )
        self.cv = nn.Sequential(*modules)
    
    def forward(self, obs: th.Tensor):
        embedded_obs: th.Tensor = self.cv(obs)
        return embedded_obs.reshape(obs.size(0), -1)

class RecurrentStateSpaceModel(nn.Module):
    def __init__(self, state_dim, action_dim, rnn_hidden_dim, hidden_dim=200, min_stddev=0.1, act=nn.ELU(inplace=True)):
        super(RecurrentStateSpaceModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc_state_action = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc_rnn_hidden = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc_state_mean_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_rnn_hidden_embedded_obs = nn.Linear(rnn_hidden_dim + 1024, hidden_dim)
        self.fc_state_mean_posterior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_posterior = nn.Linear(hidden_dim, state_dim)
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self._min_stddev = min_stddev
        self.act = act

    def forward(self, state, action, rnn_hidden, embedded_next_obs):
        next_state_prior, rnn_hidden = self.prior(state, action, rnn_hidden)
        next_state_posterior = self.posterior(rnn_hidden, embedded_next_obs)
        return next_state_prior, next_state_posterior, rnn_hidden

    def prior(self, state, action, rnn_hidden):
        hidden = self.act(self.fc_state_action(th.cat([state, action], dim=1)))
        rnn_hidden = self.rnn(hidden, rnn_hidden)
        hidden = self.act(self.fc_rnn_hidden(rnn_hidden))

        mean = self.fc_state_mean_prior(hidden)
        stddev = F.softplus(self.fc_state_stddev_prior(hidden)) + self._min_stddev
        return Normal(mean, stddev), rnn_hidden

    def posterior(self, rnn_hidden, embedded_obs):
        hidden = self.act(self.fc_rnn_hidden_embedded_obs(th.cat([rnn_hidden, embedded_obs], dim=1)))
        mean = self.fc_state_mean_posterior(hidden)
        stddev = F.softplus(self.fc_state_stddev_posterior(hidden)) + self._min_stddev
        return Normal(mean, stddev)

class ObservationModel(nn.Module):
    def __init__(self, state_dim, rnn_hidden_dim, channel=3, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.fc = nn.Linear(state_dim+rnn_hidden_dim, 1024)
        channels = [1024, 128, 64, 32, channel]
        kernel_sizes = [5, 5, 6, 6]
        stride = 2

        modules = []
        for i in range(len(channels)-1):
            if i == len(channels)-2:
                modules.append(nn.ConvTranspose2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernel_sizes[i], stride=stride))
            else:
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernel_sizes[i], stride=stride),
                        activation,
                    )
                )
        self.dc = nn.Sequential(*modules)
    
    def forward(self, state: th.Tensor, rnn_hidden: th.Tensor):
        hidden: th.Tensor = self.fc(th.cat([state, rnn_hidden], dim=1))
        hidden = hidden.view(hidden.size(0), 1024, 1, 1)
        obs = self.dc(hidden)
        return obs

class RewardModel(nn.Module):
    def __init__(self, state_dim, rnn_hidden_dim, hidden_dim=400, act=nn.ELU(inplace=True)):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim+rnn_hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, rnn_hidden):
        reward = self.fc(th.cat([state, rnn_hidden], dim=1))
        return reward

class ValuedModel(nn.Module):
    def __init__(self, state_dim, rnn_hidden_dim, hidden_dim=400, act=nn.ELU(inplace=True)):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim+rnn_hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, rnn_hidden):
        value = self.fc(th.cat([state, rnn_hidden], dim=1))
        return value

class ActionModel(nn.Module):
    def __init__(self, state_dim, rnn_hidden_dim, action_dim, hidden_dim=400, act=nn.ELU(inplace=True), min_stddev=1e-4, init_stddev=5.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim+rnn_hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act
        )
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_stddev = nn.Linear(hidden_dim, action_dim)
        self.min_stddev = min_stddev
        self.init_stddev = np.log(np.exp(init_stddev) - 1)

    def forward(self, state, rnn_hidden, training=True):
        hidden = self.fc(th.cat([state, rnn_hidden], dim=1))

        mean = self.fc_mean(hidden)
        mean = 5.0 * th.tanh(mean / 5.0)
        stddev = self.fc_stddev(hidden)
        stddev = F.softplus(stddev + self.init_stddev) + self.min_stddev

        if training:
            action = th.tanh(Normal(mean, stddev).rsample())
        else:
            action = th.tanh(mean)
        return action

class Agent:
    def __init__(self, encoder: Encoder, rssm: RecurrentStateSpaceModel, action_model: ActionModel):
        self.encoder = encoder
        self.rssm = rssm
        self.action_model = action_model

        self.device = next(self.action_model.parameters()).device
        self.rnn_hidden = th.zeros(1, rssm.rnn_hidden_dim, device=self.device)

    def __call__(self, obs, training=True):
        obs = normalize_observation(obs)
        obs = th.as_tensor(obs, device=self.device)
        obs = obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)

        with th.no_grad():
            embedded_obs = self.encoder(obs)
            state_posterior = self.rssm.posterior(self.rnn_hidden, embedded_obs)
            state = state_posterior.sample()
            action: th.Tensor = self.action_model(state, self.rnn_hidden, training=training)

            _, self.rnn_hidden = self.rssm.prior(state, action, self.rnn_hidden)

        return action.squeeze().cpu().numpy()

    def reset(self):
        self.rnn_hidden = th.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)

class Dreamer:
    def __init__(self, action_dim, channel=3, state_dim=30, rnn_hidden_dim=200, model_lr=6e-4, value_lr=8e-5, action_lr=8e-5, eps=1e-4, device='cpu'):
        self.encoder = Encoder(channel).to(device)
        self.rssm = RecurrentStateSpaceModel(state_dim, action_dim, rnn_hidden_dim).to(device)
        self.obs_model = ObservationModel(state_dim, rnn_hidden_dim).to(device)
        self.reward_model = RewardModel(state_dim, rnn_hidden_dim).to(device)
        self.value_model = ValuedModel(state_dim, rnn_hidden_dim).to(device)
        self.action_model = ActionModel(state_dim, rnn_hidden_dim, action_dim).to(device)

        self.model_params = (list(self.encoder.parameters()) + list(self.rssm.parameters()) + list(self.obs_model.parameters()) + list(self.reward_model.parameters()))
        self.model_optim = th.optim.Adam(self.model_params, lr=model_lr, eps=eps)
        self.value_optim = th.optim.Adam(self.value_model.parameters(), lr=value_lr, eps=eps)
        self.action_optim = th.optim.Adam(self.action_model.parameters(), lr=action_lr, eps=eps)

        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.device = device
    
    def update(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray, free_nats=3, clip_grad_norm=100, imagination_horizon=15, gamma=0.9, lambda_=0.95) -> Dict[str, th.Tensor]:
        flatten_states, flatten_rnn_hiddens, model_losses = self.update_model(observations, actions, rewards, free_nats, clip_grad_norm)
        action_value_losses = self.update_action_value(flatten_states, flatten_rnn_hiddens, imagination_horizon, gamma, lambda_, clip_grad_norm)
        return dict(**model_losses, **action_value_losses)

    def update_model(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray, free_nats=3, clip_grad_norm=100):
        device = self.device
        batch_size = observations.shape[0]
        chunk_length = observations.shape[1]

        observations = normalize_observation(observations)
        observations = th.as_tensor(observations, device=device)
        observations = observations.transpose(3, 4).transpose(2, 3)
        observations = observations.transpose(0, 1)
        actions = th.as_tensor(actions, device=device).transpose(0, 1)
        rewards = th.as_tensor(rewards, device=device).transpose(0, 1)

        embedded_observations = self.encoder(observations.reshape(-1, 3, 64, 64)).view(chunk_length, batch_size, -1)

        states = th.zeros(chunk_length, batch_size, self.state_dim, device=device)
        rnn_hiddens = th.zeros(chunk_length, batch_size, self.rnn_hidden_dim, device=device)

        state = th.zeros(batch_size, self.state_dim, device=device)
        rnn_hidden = th.zeros(batch_size, self.rnn_hidden_dim, device=device)

        kl_loss = 0
        for l in range(chunk_length-1):
            next_state_prior, next_state_posterior, rnn_hidden = self.rssm(state, actions[l], rnn_hidden, embedded_observations[l+1])
            state = next_state_posterior.rsample()
            states[l+1] = state
            rnn_hiddens[l+1] = rnn_hidden
            kl = kl_divergence(next_state_prior, next_state_posterior).sum(dim=1)
            kl_loss += kl.clamp(min=free_nats).mean()
        kl_loss /= (chunk_length - 1)

        states = states[1:]
        rnn_hiddens = rnn_hiddens[1:]
        flatten_states = states.view(-1, self.state_dim)
        flatten_rnn_hiddens = rnn_hiddens.view(-1, self.rnn_hidden_dim)
        recon_observations = self.obs_model(flatten_states, flatten_rnn_hiddens).view(chunk_length-1, batch_size, 3, 64, 64)
        predicted_rewards = self.reward_model(flatten_states, flatten_rnn_hiddens).view(chunk_length-1, batch_size, 1)
        obs_loss = 0.5 * F.mse_loss(recon_observations, observations[1:], reduction='none').mean([0, 1]).sum()
        reward_loss = 0.5 * F.mse_loss(predicted_rewards, rewards[:-1])

        model_loss = kl_loss + obs_loss + reward_loss
        self.model_optim.zero_grad()
        model_loss.backward()
        clip_grad_norm_(self.model_params, clip_grad_norm)
        self.model_optim.step()

        losses = {
            'model_loss': model_loss,
            'kl_loss': kl_loss,
            'obs_loss': obs_loss,
            'reward_loss': reward_loss,
        }

        return flatten_states, flatten_rnn_hiddens, losses

    def update_action_value(self, flatten_states: th.Tensor, flatten_rnn_hiddens: th.Tensor, imagination_horizon=15, gamma=0.9, lambda_=0.95, clip_grad_norm=100):
        flatten_states = flatten_states.detach()
        flatten_rnn_hiddens = flatten_rnn_hiddens.detach()

        imaginated_states = th.zeros(imagination_horizon + 1, *flatten_states.shape, device=flatten_states.device)
        imaginated_rnn_hiddens = th.zeros(imagination_horizon + 1, *flatten_rnn_hiddens.shape, device=flatten_rnn_hiddens.device)

        for h in range(1, imagination_horizon + 1):
            actions = self.action_model(flatten_states, flatten_rnn_hiddens)
            flatten_states_prior, flatten_rnn_hiddens = self.rssm.prior(flatten_states, actions, flatten_rnn_hiddens)
            flatten_states = flatten_states_prior.rsample()
            imaginated_states[h] = flatten_states
            imaginated_rnn_hiddens[h] = flatten_rnn_hiddens

        flatten_imaginated_states = imaginated_states.view(-1, self.state_dim)
        flatten_imaginated_rnn_hiddens = imaginated_rnn_hiddens.view(-1, self.rnn_hidden_dim)
        imaginated_rewards = self.reward_model(flatten_imaginated_states, flatten_imaginated_rnn_hiddens).view(imagination_horizon + 1, -1)
        imaginated_values = self.value_model(flatten_imaginated_states, flatten_imaginated_rnn_hiddens).view(imagination_horizon + 1, -1)
        lambda_target_values = lambda_target(imaginated_rewards, imaginated_values, gamma, lambda_)

        action_loss = -1 * (lambda_target_values.mean())
        self.action_optim.zero_grad()
        action_loss.backward()
        clip_grad_norm_(self.action_model.parameters(), clip_grad_norm)
        self.action_optim.step()

        imaginated_values = self.value_model(flatten_imaginated_states.detach(), flatten_imaginated_rnn_hiddens.detach()).view(imagination_horizon + 1, -1)
        lambda_target_values = lambda_target(imaginated_rewards.detach(), imaginated_values, gamma, lambda_)

        value_loss = 0.5 * F.mse_loss(imaginated_values, lambda_target_values.detach())
        self.value_optim.zero_grad()
        value_loss.backward()
        clip_grad_norm_(self.value_model.parameters(), clip_grad_norm)
        self.value_optim.step()

        losses = {
            'action_loss': action_loss,
            'value_loss': value_loss,
        }

        return losses
    
    def state_dict(self):
        return {
            'encoder': self.encoder.state_dict(),
            'rssm': self.rssm.state_dict(),
            'obs_model': self.obs_model.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'value_model': self.value_model.state_dict(),
            'action_model': self.action_model.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict['encoder'])
        self.rssm.load_state_dict(state_dict['rssm'])
        self.obs_model.load_state_dict(state_dict['obs_model'])
        self.reward_model.load_state_dict(state_dict['reward_model'])
        self.value_model.load_state_dict(state_dict['value_model'])
        self.action_model.load_state_dict(state_dict['action_model'])
    
    def eval(self):
        self.encoder.eval()
        self.rssm.eval()
        self.obs_model.eval()
        self.reward_model.eval()
        self.value_model.eval()
        self.action_model.eval()
    
    def train(self):
        self.encoder.train()
        self.rssm.train()
        self.obs_model.train()
        self.reward_model.train()
        self.value_model.train()
        self.action_model.train()
    
    def save(self, path):
        th.save(self.state_dict(), path)
    
    def load(self, path):
        state_dict = th.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
