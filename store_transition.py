
import torch
import numpy as np
import random

class Store:
    def __init__(self, mem_size , state_dim, act_dim, scale,max_length=None, max_ep_len=3678,):

        self.mem_size = mem_size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_ep_len = max_ep_len
        self.max_length = max_length
        self.scale = scale
        self.device = "cuda"

        self.states = torch.empty((self.mem_size, self.state_dim, self.state_dim),dtype=torch.uint8)
        self.actions = torch.empty(self.mem_size, self.act_dim , dtype=torch.int32)
        self.rewards = torch.empty((self.mem_size), dtype=torch.int32)
        self.count = 0
        self.current = 0
#=====

    def store_transition(self, state, a, r):  # 存储状态信息
        if not hasattr(self, 'memory_counter'):  # hasattr() 函数用于判断对象是否包含对应的属性。
            self.memory_counter = 0
        self.states[self.current] = torch.tensor(state)  # 将状态转换为torch张量
        self.actions[self.current] = torch.tensor(a)
        self.rewards[self.current] = torch.tensor(r)
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.mem_size

    def discount_cumsum(self,x, gamma):  # 本时间的权重等于下个时间的权重乘衰减系数gamma再加上该时刻的单步奖励。
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def get_batch(self, batch_size=16, max_len=20):
        # state_mean, state_std = self.get_mean_std()
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            j = np.random.randint(0, 3078)
            s.append(self.states[j].reshape(1, -1, self.state_dim))
            a.append(self.actions[j].reshape(1, -1, self.act_dim))
            while True:
                if a[-1].shape[1] < s[-1].shape[1]:
                    a[-1] = np.concatenate([a[-1], np.zeros((1, 1, a[-1].shape[2]))], axis=1)
                else:
                    break
            r.append(self.rewards[j].reshape(1, -1, 1))
            si = random.randint(0, len(self.rewards) - 1)
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff填充截断
            rtg.append(
                self.discount_cumsum(self.rewards[si:si + s[-1].shape[1]], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            while True:
                if rtg[-1].shape[1] < s[-1].shape[1]:
                    rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
                else:
                    break
            tlen = s[-1].shape[1]
            if tlen > max_len:
                tlen = max_len
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, self.state_dim)), s[-1]], axis=1)
            # s[-1] = (s[-1] - state_mean) / state_std
            # a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)  #待定
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)

            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, self.state_dim))], axis=1))

        # torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32)
        # print(f's:{s}')
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32)
        # print(f'a:{a}')
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32)
        # print(f'r:{r}')
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32)
        # print(f'rtg:{rtg}')
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long)
        mask = torch.from_numpy(np.concatenate(mask, axis=0))

        s = s.reshape(1, -1, self.state_dim)
        a = a.reshape(1, -1, self.act_dim)
        rtg = (rtg).reshape(1, -1, 1)
        r = (r).reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        mask = mask.reshape(1, -1)

        s = s[:, -self.max_length:]
        a = a[:, -self.max_length:]
        rtg = rtg[:, -self.max_length:]
        r = r[:, -self.max_length:]
        timesteps = timesteps[:, -self.max_length:]
        mask = mask[:, -self.max_length:]
        # return s, a, r, d, rtg, timesteps, mask

        return s, a, r, rtg, timesteps, mask
