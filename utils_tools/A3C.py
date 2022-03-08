# -*- coding: utf-8 -*-
import torch
from models import ac_net
from torch.distributions import Normal
from collections import deque
from copy import deepcopy
import torch.multiprocessing as mp
import numpy as np


T_MAX = 5

class AC_Net:
    def __int__(self, state_dim, name, action_bound):
        self.actionbound = action_bound
        self.state_dim = state_dim
        self.name = name
        self.AC = ac_net(inner_dim=self.state_dim)
        self.t_max = T_MAX
        self.memory = deque(maxlen=2*T_MAX)
        self.decay_index = 0.99
        self.t = 0

    def get_action(self, feature):
        mu, sigma, _ = self.AC(feature)
        distribution = Normal(mu.detach(), sigma.detach() + 1e-8)       # 1e-8防止数据上溢
        action = torch.clamp(distribution.sample(), self.actionbound.min(), self.actionbound.max())     # 将输出clamp到动作空间
        return action

    def state_store_memory(self, state, action, reward, done):
        self.memory.append((state, action, reward, done, self.t))

    def loss(self):
        state, action, reward, done = zip(*self.memory)
        state = np.stack(state, axis=0).squeeze()
        action = np.concatenate(action).reshape(self.t_max, -1)
        reward = np.concatenate(reward).reshape(self.t_max, -1)
        done = np.array(done, dtype='float32').reshape(self.t_max, -1)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        # 使用网络生成正态分布的mu和sigma，建立分布之后用状态序列中的动作计算在该分布上的log_prob
        mu, sigma, _ = self.AC(state[-1, ...])
        distrib = Normal(mu.detach(), sigma.detach() + 1e-8)
        log_p_action = distrib.log_prob(action)

        # 优势函数计算
        if done[-1, ...] == 1:
            d_reward = self.decayed_reward(state[-1, ...], reward, True)
        else:
            d_reward = self.decayed_reward(state[-1, ...], reward, False)
        adv = self.advantage_cal(state, d_reward)

        # entropy计算？？？

        # loss计算
        actor_loss = - log_p_action * adv
        critic_loss = torch.pow(adv, 2)
        loss_total = (actor_loss + critic_loss).mean()
        return loss_total

    def advantage_cal(self, state_seq, q_val_estimate):     # q_val_estimate: 动作价值函数近似,等价于最后一个状态价值按状态序列reward向前推
        _, _, value = self.AC(state_seq)
        q_val_estimate = torch.FloatTensor(q_val_estimate)
        target_func = q_val_estimate - value
        return target_func

    # 将以此计算中可以得到的mu，sigma，value计算拆分成两次计算，增加计算量，需改进
    def decayed_reward(self, last_step_state, reward_seq, terminate=None):
        decayed_rd = []
        # 终止状态下回报函数计算
        if terminate:
            value_target = torch.tensor(0.0)
        else:
            state_frame = torch.Tensor(last_step_state)
            _, _, value_target = self.AC(state_frame).detach().numpy()
        for rd_ in reward_seq[::-1]:
            value_target = rd_ + value_target * self.decay_index
            decayed_rd.append(value_target)
        decayed_rd.reverse()
        return decayed_rd

    def save_model(self, file_name):
        checkpoint = {'actor': self.AC.state_dict()}
        torch.save(checkpoint, file_name)

    def load_model(self, file_name):
        checkpoint = torch.load(file_name)
        self.AC.load_state_dict(checkpoint['model'])

class worker(mp.Process):
    def __init__(self, name):
        super(worker, self).__init__()
        pass

    def run(self):
        pass
        


