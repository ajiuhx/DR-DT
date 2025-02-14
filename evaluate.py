import numpy as np
import torch

import time

from maze_env import MAZE_H, Maze


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
env = Maze()
max_ep_len = 6000
# env_targets = [7200, 3600]  # evaluation conditioning targets评估条件目标
env_targets = 1000 # evaluation conditioning targets评估条件目标
scale = 1000.

# state_dim = env.n_features
state_dim = MAZE_H
act_dim = env.n_actions
class Test:

    def __init__(self, model, scheduler=None, eval_fns=None):
        self.model = model
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        logs = dict()
        # self.model.load_state_dict(torch.load('../trained_model.pth'))

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs
