import numpy as np
import torch

import time

from maze_env import MAZE_H,Maze


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
env = Maze()
max_ep_len = 15000
# env_targets = [7200, 3600]  # evaluation conditioning targets评估条件目标
env_targets = 1000 # evaluation conditioning targets评估条件目标
scale = 1000.

# state_dim = env.n_features
state_dim = MAZE_H
act_dim = env.n_actions
class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch,loss_fn,epoch, scheduler=None, eval_fns=None):#state_mean, state_std,
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        # self.state_mean = state_mean,
        # self.state_std = state_std,

        self.start_time = time.time()
        self.epoch = epoch

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()
        i=0
        j=0
        self.model.train()
        torch.cuda.empty_cache()
        for _ in range(num_steps):
            train_loss = self.train_step()
            # print(train_loss)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()
            # print(i)
            i = i+1
            j=j+1
        # torch.save(self.model.state_dict(), '../trained_model.pth')

        logs['time/training'] = time.time() - train_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        del self.model
        torch.cuda.empty_cache()

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
