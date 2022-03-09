# -*- coding: utf-8 -*-
from copy import deepcopy


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
def record(global_ep, global_ep_r, ep_r, res_queue, worker_ep, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)

    print(f'{name}, '
          f'Global_EP: {global_ep.value}, '
          f'worker_EP: {worker_ep}, '
          f'EP_r: {global_ep_r.value}, '
          f'reward_ep: {ep_r}')
