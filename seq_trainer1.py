import torch
from trainer import Trainer
from maze_env import MAZE_H, MAZE_W
from maze_env import Maze
from store_transition import Store

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
env = Maze()
max_ep_len = 15000
# env_targets = [7200, 3600]  # evaluation conditioning targets评估条件目标
env_targets = 1800 # evaluation conditioning targets评估条件目标
scale = 1000.

state_dim = MAZE_H
act_dim =env.n_actions


store = Store(state_dim=state_dim,
              act_dim=act_dim,
              max_length=20,
              max_ep_len=max_ep_len,
              scale = scale,
              mem_size=20000)
class SequenceTrainer(Trainer):
    cost_his = []
    def train_step(self, act_dim=None, num_params=None, cost_his=None):
        # # 初始化TensorBoard写入器
        # writer = SummaryWriter("../logs/")
        for t in range(self.epoch):
            states, actions_, rewards, rtg, timesteps, attention_mask = store.get_batch(self.batch_size,20)
            # states, actions_, rewards, rtg, timesteps, attention_mask = self.get_batch

            action_target = torch.clone(actions_)
            state_target = torch.clone(states)
            reward_target = torch.clone(rewards)
            action_target = action_target.to(device)
            state_target = state_target.to(device)
            reward_target = reward_target.to(device)
            state_preds, action_preds, reward_preds = self.model.forward(
                state_target, action_target, reward_target, rtg.to(device), timesteps.to(device), attention_mask=attention_mask.to(device),
            )

            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

            global loss
            loss = self.loss_fn(
                state_preds, action_preds, reward_preds,
                state_target, action_target, reward_target,
            )
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            # 更新参数
            self.optimizer.step()

            print("epoch[{}/{}]===>loss:{} ".format(t,self.epoch,loss))

            with torch.no_grad():
                self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
                # 关闭写入器
                # writer.close()
        # self.model.save_model(self.model, self.optimizer)


        return loss.detach().cpu().item()
