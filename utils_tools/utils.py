# -*- coding: utf-8 -*-
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gradient_push(local_net, global_net):
    '''
    paramter._grad与parameter.grad指向内存不同
    id(self.ac_model.AC.critic1.weight.grad[0])
    >>>1520533070616
    id(self.ac_model.AC.critic1.weight._grad[0])
    >>>1520532354536
    需要查阅官方文档
    '''
    for loc, glo in zip(local_net.parameters(), global_net.parameters()):
        glo._grad = loc.grad


def pull_weight(local_net, global_net):
    ckpt = global_net.state_dict()
    local_net.load_state_dict(ckpt)


# record function
'''
from 'Simple implementation of Reinforcement Learning (A3C) using Pytorch'
by: MorvanZhou
link: https://github.com/MorvanZhou/pytorch-A3C
'''
def record(global_ep, global_ep_r, ep_r, res_queue, worker_ep, name, idx):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put([idx, ep_r])

    print(f'{name}, '
          f'Global_EP: {global_ep.value}, '
          f'worker_EP: {worker_ep}, '
          f'EP_r: {global_ep_r.value}, '
          f'reward_ep: {ep_r}')


def smooth_tsplot(dict_data, thread_num):
    time_seq = []
    for keys in dict_data.keys():
        dict_data[keys] = np.hstack(dict_data[keys])
        time_seq.append(dict_data[keys])
    # 将原数据转置为(episode, worker)格式
    time_seq = np.vstack(time_seq).T
    # 使数据列连续
    time_seq = np.ascontiguousarray(time_seq)
    # 创建dataframe，设置列索引
    df = pd.DataFrame(time_seq, columns=[f'worker{idx}' for idx in range(thread_num)])

    # 插入episodes数据
    df.insert(0, 'episodes', [i for i in range(df.shape[0])])
    # 将pandas宽数据转为长数据，将多列数据合并为一列，生成新的dataframe
    df_long = pd.melt(df, 'episodes', var_name='worker', value_name='ep_reward')
    df_long.head()
    # 以长格式模式传递整个数据集将聚合重复值以显示平均值和 95% 置信区间显示
    sns.lineplot(data=df_long, x='episodes', y='ep_reward')
    plt.show()
