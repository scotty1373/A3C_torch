# -*- coding:utf-8 -*-
import sys
import os
import time
import torch
from utils_tools.A3C import AC_Net, worker
import torch.multiprocessing as mp


STATE_DIM = 3
ACTION_BOUND = 2
THREAD_NUM = 5


if __name__ == '__main__':
    global_net = AC_Net(STATE_DIM, ACTION_BOUND)
    # global_net.AC.share_memory()
    g_opt = torch.optim.Adam(params=global_net.AC.parameters(), lr=1e-4)

    # 创建共享内存和跨进程通信队列
    global_ep = mp.Value('i', 0)    # C_Type int类型共享内训
    global_r = mp.Value('f', 0)
    res_queue = mp.Queue()

    # 从自定义进程类worker中创建子线程
    p_list = [worker(f'{idx}_worker', global_net, g_opt, global_ep, global_r, res_queue) for idx in range(THREAD_NUM)]
    [p.start() for p in p_list]

    res = []
    while True:
        r_ep = res_queue.get()
        if r_ep:
            res.append(r_ep)
        else:
            break
    [p.join() for p in p_list]

    time.time()

