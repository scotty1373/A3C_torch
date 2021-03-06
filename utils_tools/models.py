# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Actor_Model(torch.nn.Module):
    def __init__(self, inner_shape, hidden_layers, outer_shape, tanh=None):
        super(Actor_Model, self).__init__()
        self.inner_shape = inner_shape
        self.outer_shape = outer_shape
        self.hidden_layers = hidden_layers      # 中间隐藏层结构
        layers = []
        for idx, hidden_unit in enumerate(self.hidden_layers):
            if not idx:
                layers.append(nn.Linear(self.inner_shape, hidden_unit))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Linear(self.hidden_layers[idx - 1], hidden_unit))
                layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(self.hidden_layers[-1], self.outer_shape))
        if tanh:
            layers.append(nn.Tanh())
        self.net_seq = nn.Sequential(*layers)

    def forward(self, input_tensor):
        return self.net_seq(input_tensor)


class Critic_Model(torch.nn.Module):
    def __init__(self, state_dim, hidden_layers=None):
        super(Critic_Model, self).__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128]
        assert isinstance(state_dim, int), 'state_dim data format error'

        self.inner_shape = state_dim
        self.hidden_layers = hidden_layers
        layers = []
        for idx, hidden_unit in enumerate(self.hidden_layers):
            if not idx:
                layers.append(nn.Linear(self.inner_shape, hidden_unit))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Linear(self.hidden_layers[idx - 1], hidden_unit))
                layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(self.hidden_layers[-1], 1))
        self.net_seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.net_seq(x)


class ac_net(torch.nn.Module):
    def __init__(self, inner_dim):
        super(ac_net, self).__init__()
        self.inner_dim = inner_dim
        self.actor1 = nn.Linear(inner_dim, 400)
        self.mu = nn.Linear(400, 1)
        self.muact = nn.Tanh()
        self.sigma = nn.Linear(400, 1)
        self.critic1 = nn.Linear(inner_dim, 300)
        self.value = nn.Linear(300, 1)
        for layer in [self.actor1, self.mu, self.sigma, self.critic1, self.value]:
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, input_tensor):
        a1 = F.relu(self.actor1(input_tensor))
        mu = self.muact(self.mu(a1)) * 2
        sigma = F.softplus(self.sigma(a1))
        c1 = F.relu(self.critic1(input_tensor))
        value = self.value(c1)
        return mu, sigma, value


class ac_net_lstm(torch.nn.Module):
    def __init__(self, inner_dim):
        super(ac_net_lstm, self).__init__()
        self.inner_dim = inner_dim
        self.lstm = nn.LSTM(3, 128, 1)
        self.actor1 = nn.Linear(128, 64)
        self.mu = nn.Linear(64, 1)
        self.muact = nn.Tanh()
        self.sigma = nn.Linear(64, 1)
        self.critic1 = nn.Linear(128, 32)
        self.value = nn.Linear(32, 1)
        for layer in [self.actor1, self.mu, self.sigma, self.critic1, self.value]:
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, input_tensor):
        lstm_common, _ = self.lstm(input_tensor.unsqueeze(1))
        a1 = F.relu(self.actor1(lstm_common.squeeze()))
        mu = self.muact(self.mu(a1)) * 2
        sigma = F.softplus(self.sigma(a1))
        c1 = F.relu(self.critic1(lstm_common.squeeze()))
        value = self.value(c1)
        return mu, sigma, value


if __name__ == '__main__':
    model = ac_net_lstm(inner_dim=3)
    x = torch.randn((10, 1, 3))
    out = model(x)
    print(f'{m}, {s}, {v}')
    model.share_memory()



