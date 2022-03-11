# -*- coding: utf-8 -*-
import math

import gym
import torch
from .models import ac_net, ac_net_lstm
from torch.distributions import Normal
from collections import deque
import torch.multiprocessing as mp
import numpy as np
from .utils import gradient_push, pull_weight, record


T_MAX = 5
STATE_DIM = 3
ACTION_BOUND = 2
MAX_TIMESTEP = 2000
MAX_EPISODE = 50

class AC_Net:
    def __init__(self, state_dim, action_bound):
        self.actionBound = action_bound
        self.state_dim = state_dim
        self.AC = ac_net(inner_dim=self.state_dim)
        self.t_max = T_MAX
        self.memory = deque(maxlen=2*T_MAX)
        self.decay_index = 0.9
        self.t = 0

    def get_action(self, feature):
        # 判断进来的数据是否为单精度张量
        if not isinstance(feature, torch.FloatTensor):
            feature = torch.FloatTensor(feature)
        mu, sigma, _ = self.AC(feature)
        distribution = Normal(mu.detach(), sigma.detach() + 1e-8)       # 1e-8防止数据上溢
        action = torch.clamp(distribution.sample(), -self.actionBound, self.actionBound)     # 将输出clamp到动作空间
        return action

    def state_store_memory(self, state, action, reward, done):
        self.memory.append((state, action, reward, done, self.t))

    def loss(self, state_t1):
        state, action, reward, done, _ = zip(*self.memory)
        state = np.stack(state, axis=0).squeeze()
        action = np.concatenate(action).reshape(self.t_max, -1)
        reward = np.concatenate(reward).reshape(self.t_max, -1)
        done = np.array(done, dtype='float32').reshape(self.t_max, -1)

        # n step尾部序列
        state_t1 = torch.FloatTensor(state_t1)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)

        # 使用网络生成正态分布的mu和sigma，建立分布之后用状态序列中的动作计算在该分布上的log_prob
        mu, sigma, _ = self.AC(state)
        distrib = Normal(mu, sigma + 1e-8)
        log_p_action = distrib.log_prob(action)

        # 优势函数计算
        if done[-1, ...] == 1:
            d_reward = self.decayed_reward(state_t1, reward, True)
        else:
            d_reward = self.decayed_reward(state_t1, reward, False)
        d_reward = np.vstack(d_reward)
        adv = self.advantage_cal(state, d_reward)

        # entropy计算？？？
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(distrib.scale)
        # loss计算
        actor_loss = - (log_p_action * adv.detach() + 0.005 * entropy)

        # 防止critic数值过大导致共享网络层更新错误
        critic_loss = torch.pow(adv, 2) * 0.2
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
            _, _, value_target = self.AC(state_frame)
            value_target = value_target.detach().data.numpy()

        '''
        ValueError: step must be greater than zero
        解决方法：https://blog.csdn.net/weixin_42716570/article/details/113957337
        '''
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
    def __init__(self, name, g_net, g_opt, global_ep, global_r, global_res, worker_id):
        super(worker, self).__init__()
        self.workerID = worker_id
        self.name = f'{name}'
        self.ac_model = AC_Net(STATE_DIM, ACTION_BOUND)
        self.g_net = g_net
        self.g_opt = g_opt
        self.global_ep, self.global_r, self.global_res = global_ep, global_r, global_res
        self.env = gym.make('Pendulum-v0').unwrapped
        # 固定初始化种子
        # self.env.seed(1)

    def run(self):
        worker_ep = 0
        while worker_ep < MAX_EPISODE:
            obs = self.env.reset()
            obs = obs.reshape(1, 3)
            # 清空n-step状态保存序列
            self.ac_model.memory.clear()

            ep_r = 0
            for idx in range(MAX_TIMESTEP):
                self.env.render()
                act_sample = self.ac_model.get_action(obs)
                # 将get_action输出的动作采样值截断梯度转为np。array类型，否则gym计算中会带有tensor类型
                obs_t1, reward, done, _ = self.env.step(act_sample.detach().numpy().reshape(1, 1))
                obs_t1 = obs_t1.reshape(1, 3)
                # reward归一化
                reward = (reward + 8) / 8

                ep_r += reward
                self.ac_model.state_store_memory(obs, act_sample.detach().numpy().reshape(1, 1), reward, done)
                if (idx+1) % self.ac_model.t_max == 0 or done:
                    loss_mean = self.ac_model.loss(obs_t1)
                    self.g_opt.zero_grad()
                    loss_mean.backward()
                    # 上传本地计算gradienr到全局参数
                    gradient_push(self.ac_model.AC, self.g_net.AC)

                    self.g_opt.step()

                    # global_net pull paramters
                    pull_weight(self.ac_model.AC, self.g_net.AC)

                    # 清空状态序列缓存
                    self.ac_model.memory.clear()

                    if done:
                        record(self.global_ep, self.global_r, ep_r, self.global_res, self.name, self.workerID)
                        break

                obs = obs_t1
            worker_ep += 1
            record(self.global_ep, self.global_r, ep_r, self.global_res, worker_ep, self.name, self.workerID)
        # 训练完成，主线程接收None数据队列空
        self.global_res.put(None)


















        


