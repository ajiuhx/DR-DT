import argparse
import random
import numpy as np
import torch
from decision_transformer import DecisionTransformer
from evaluate_episodes import evaluate_episode_rtg
from seq_trainer1 import SequenceTrainer
from maze_env import Maze


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=20, help="数据标准化")
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_type', type=str, default='dt', help="模型类型")
    parser.add_argument('--embed_dim', type=int, default=252)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-1)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=1)
    parser.add_argument('--num_steps_per_iter', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--state_dim', type=int, default=84)
    parser.add_argument('--act_dim', type=int, default=6300)
    parser.add_argument('--max_ep_len', type=int, default=3678)
    parser.add_argument('--scale', type=float, default=1000.)
    parser.add_argument('--env_targets', type=float, default=3040)
    parser.add_argument('--epoch', type=int, default=2000, help="训练次数")
    parser.add_argument('--states_path', type=str, default='./states.pt', help="状态数据参数路径")
    parser.add_argument('--rewards_path', type=str, default='./rewards.pt', help="状态数据参数路径")
    parser.add_argument('--actions_path', type=str, default='./actions.pt', help="状态数据参数路径")

    return vars(parser.parse_args())


# 本时间的权重等于下个时间的权重乘衰减系数gamma再加上该时刻的单步奖励。
def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def get_batch(batch_size, max_len, states, state_dim,
              act_dim, returns, actions, max_ep_len, state_mean, state_std, scale, device):
    s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
    for i in range(batch_size):
        j = np.random.randint(0, 1970)
        s.append(states[j].reshape(1, -1, state_dim))
        a.append(actions[j].reshape(1, -1, act_dim))
        while True:
            if a[-1].shape[1] < s[-1].shape[1]:
                a[-1] = np.concatenate([a[-1], np.zeros((1, 1, a[-1].shape[2]))], axis=1)
            else:
                break
        r.append(returns[j].reshape(1, -1, 1))
        si = random.randint(0, len(returns) - 1)
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff填充截断
        rtg.append(discount_cumsum(returns[si:si + s[-1].shape[1]], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
        while True:
            if rtg[-1].shape[1] < s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
            else:
                break
        tlen = s[-1].shape[1]
        if tlen > max_len:
            tlen = max_len
        s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
        s[-1] = (s[-1] - state_mean) / state_std
        # a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)  #待定
        r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
        rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
        timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)

        mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, state_dim))], axis=1))

    # torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    # print(f's:{s}')
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    # print(f'a:{a}')
    r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
    # print(f'r:{r}')
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
    # print(f'rtg:{rtg}')
    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    # return s, a, r, d, rtg, timesteps, mask

    return s, a, r, rtg, timesteps, mask


def eval_episodes(target_rew, num_eval_episodes, state_dim, act_dim, max_ep_len, scale, mode, state_mean, state_std,
                  device):
    def fn(model):
        returns, lengths = [], []
        for _ in range(num_eval_episodes):
            # 包装器“ with torch.no_grad（）”将所有require_grad标志临时设置为false。
            # 不希望PyTorch计算新定义的变量param的梯度（减少计算量），因为他只想更新它们的值。
            with torch.no_grad():
                ret, length = evaluate_episode_rtg(
                    Maze(),
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rew / scale,
                    mode=mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                )
            returns.append(ret)
            lengths.append(length)
    return fn


def load_states_actions_rewards(states_path, rewards_path, actions_path):
    states = [row for row in torch.stack(torch.load(states_path)).cpu().numpy()]
    rewards = np.array(torch.load(rewards_path))
    actions = [item.view(1, item.shape[0]).cpu().detach().numpy() for item in torch.load(actions_path)]
    states1 = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states1, axis=0), np.std(states1, axis=0) + 1e-6
    return states, rewards, actions, state_mean, state_std


def mytrain(config):
    device = config['device']
    model_type = config['model_type']
    K = config['K']
    batch_size = config['batch_size']
    epoch = config['epoch']
    state_dim = config['state_dim']
    act_dim = config['act_dim']
    max_ep_len = config['max_ep_len']
    scale = config['scale']
    warmup_steps = config['warmup_steps']
    env_targets = config['env_targets']
    max_iters = config['max_iters']
    states_path = config['states_path']
    rewards_path = config['rewards_path']
    actions_path = config['actions_path']

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=config['embed_dim'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_inner=4 * config['embed_dim'],
        activation_function=config['activation_function'],
        n_positions=1024,
        resid_pdrop=config['dropout'],
        attn_pdrop=config['dropout'],
        is_train=True,
        device='cuda',
        mem_size=20000,
        scale=1000
    )
    model = model.to(device=device)

    states, returns, actions, state_mean, state_std = load_states_actions_rewards(states_path, rewards_path,
                                                                                  actions_path)

    batch_returns = get_batch(batch_size, K, states, state_dim, act_dim, returns, actions,
                              max_ep_len, state_mean, state_std, scale, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    # LambdaLR也就可以理解成自定义规则去调整网络的学习率。从另一个角度理解，数学中的 λ \lambda λ一般是作为系数使用，
    # 因此这个学习率调度器的作用就是将初始学习率乘以人工规则所生成的系数 λ \lambda λ。

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        state_mean=state_mean,
        state_std=state_std,
        get_batch=batch_returns,
        scheduler=scheduler,
        # loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2) + torch.mean((r_hat - r) ** 2),
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        epoch=epoch,
        # eval_fns=[eval_episodes(env_targets)],
    )

    outputs = trainer.train_iteration(num_steps=config['num_steps_per_iter'], print_logs=True)
    print(outputs)


if __name__ == '__main__':
    config = load_config()
    mytrain(config)
