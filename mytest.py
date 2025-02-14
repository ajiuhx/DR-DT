import numpy as np
import torch
from decision_transformer import DecisionTransformer
from evaluate_episodes import evaluate_episode_rtg
import argparse
from maze_env import Maze
from evaluate import Test


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=20, help="数据标准化")
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_type', type=str, default='dt', help="模型类型")
    parser.add_argument('--embed_dim', type=int, default=252)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=1)
    parser.add_argument('--num_steps_per_iter', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--state_dim', type=int, default=84)
    parser.add_argument('--act_dim', type=int, default=6300)
    parser.add_argument('--max_ep_len', type=int, default=3678)
    parser.add_argument('--scale', type=float, default=1000.)
    parser.add_argument('--env_targets', type=float, default=1000)
    parser.add_argument('--epoch', type=int, default=20, help="训练次数")
    parser.add_argument('--states_path', type=str, default='./states.pt', help="状态数据参数路径")
    parser.add_argument('--rewards_path', type=str, default='./rewards.pt', help="状态数据参数路径")
    parser.add_argument('--actions_path', type=str, default='./actions.pt', help="状态数据参数路径")
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    return vars(parser.parse_args())


def load_mean_std(states_path, rewards_path, actions_path):
    states = [row for row in torch.stack(torch.load(states_path)).cpu().numpy()]
    rewards = np.array(torch.load(rewards_path))
    actions = [item.view(1, item.shape[0]).cpu().detach().numpy() for item in torch.load(actions_path)]
    states1 = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states1, axis=0), np.std(states1, axis=0) + 1e-6
    return state_mean, state_std


def eval_episodes(target_rew, num_eval_episodes, state_dim, act_dim, max_ep_len, scale, mode, state_mean, state_std,
                  device):
    def fn(model):
        returns, lengths = [], []
        for _ in range(num_eval_episodes):
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


def mytest(config):
    device = config['device']
    K = config['K']
    state_dim = config['state_dim']
    act_dim = config['act_dim']
    max_ep_len = config['max_ep_len']
    env_targets = config['env_targets']
    states_path = config['states_path']
    rewards_path = config['rewards_path']
    actions_path = config['actions_path']
    num_eval_episodes = config['num_eval_episodes']
    scale = config['scale']
    mode = config.get('mode', 'normal')

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
        mem_size=20000,
        is_train=False,
        scale = 1000,
        device='cpu'
    )

    state_mean, state_std = load_mean_std(states_path, rewards_path, actions_path)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    tester = Test(
        model=model,
        # state_mean=state_mean,
        # state_std=state_std,
        eval_fns=[eval_episodes(env_targets, num_eval_episodes, state_dim, act_dim, max_ep_len, scale, mode, state_mean,
                                state_std,
                                device)],
    )

    model.load_model(model, optimizer)
    model = model.to(device=device)
    model.eval()

    outputs = tester.train_iteration(num_steps=config['num_steps_per_iter'], print_logs=True)
    print(outputs)


if __name__ == '__main__':
    config = load_config()
    mytest(config)
