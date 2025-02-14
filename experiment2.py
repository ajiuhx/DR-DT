
# encoding: utf-8
# import gym
import numpy as np
import torch
#torch.cuda.empty_cache()
#torch.cuda.current_device()
#torch.cuda._initialized = True
from PIL import Image
import sys
sys.path.append('..')
from maze_env import MAZE_H, MAZE_W
import time
from rigid_transform_3D import rigid_transform_3D

import argparse
from evaluate_episodes import evaluate_episode_rtg
from decision_transformer2 import DecisionTransformer
from maze_env import Maze
from seq_trainer1 import SequenceTrainer
from evaluate import Test
import os
from torch.utils.tensorboard import SummaryWriter
from store_transition import Store
os.environ['LD_LIBRARY_PATH']='$LD_LIBRARY_PATH:/home/hx/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH:/usr/lib/nvidia'
# CUDA_VISIBLE_DEVICES= 0
# 指定使用一块GPU，[0,1,2,3]下面是只选择使用1号GPU
print("是否可用：", torch.cuda.is_available())        # 查看GPU是否可用
print("GPU数量：", torch.cuda.device_count())        # 查看GPU数量
print("torch方法查看CUDA版本：", torch.version.cuda)  # torch方法查看CUDA版本

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(torch.cuda.device_count())
#检测一下cuda是否可用
print(torch.cuda.is_available())
torch.backends.cudnn.enabled = False
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")
# torch.cuda.empty_cache()
def discount_cumsum(x, gamma):       #本时间的权重等于下个时间的权重乘衰减系数gamma再加上该时刻的单步奖励。
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

#experiment('gym-experiment', variant=vars(args))
num_frames = 0
num_episodes = 1
true_traj_count = 0
true_traj = []

env = Maze()
max_ep_len = 6000
env_targets = 3040 # evaluation conditioning targets评估条件目标
scale = 1000.

# state_dim = env.n_features
state_dim = MAZE_H
act_dim = env.n_actions

global ground, ground_temp, estimate_temp
global estimate
# MH01
ground = np.array([[4.687579000000000384e+00, -1.786059000000000063e+00, 8.035400000000000320e-01]])
estimate = np.array([[-0.013096054, 0.148018956, 0.051851347]])
# MH02
# ground = np.array([[4.620760999999999896e+00, -1.836673999999999918e+00, 7.462400000000000144e-01]])
# estimate = np.array([[0.079394445, -0.285317361, -0.085732698]])
#MH03
# ground = np.array([[4.637800000000000367e+00, -1.784734000000000043e+00, 5.946029999999999927e-01]])
# estimate = np.array([[0.004397152, -0.022089170, -0.012676725]])
#MH04
# ground = np.array([[4.681549000000000404e+00, -1.751622999999999930e+00, 6.035420000000000229e-01]])
# estimate = np.array([[-0.005734304, -0.027349519, -0.008012845]])
# 初始化TensorBoard写入器
writer = SummaryWriter("../logs/")
K = 'K'    # 20
mem_size = 20000

f_true = open('/home/hx/ORB_SLAM3/truth_deal/true_m1.txt', 'r')
# f_true = open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/truth_deal/true_m1.txt', 'r')
while True:
    line_true = f_true.readline()  # 该方法每次读出一行内容，返回一个字符串
    true_traj.append(line_true)
    if not line_true:
        break
f_true.close()


def get_observation(done=False):
    global num_frames
    global f
    while True:
        num = 0
        try:
            # f = open('/home/hx/ORB_SLAM3/result.txt', 'r')
            f = open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/result.txt', 'r')
            line = f.readline()  # 该方法每次读出一行内容，返回一个字符串
            line = line.strip()  # str.strip()就是把字符串(str)的头和尾的空格，以及位于头尾的\n \t之类给删掉。
            fields = line.split(' ')  # 拆分字符串。通过指定分隔符对字符串进行切片，并返回分割后的字符串列表（list）
            if len(fields[0]) > 0:
                num = int(fields[0])
        finally:
            if f:
                f.close()
        if num > num_frames:
            num_frames = num
            print('num_frames:', num_frames)
            break
        # print num, num_frames, 'wait for a new frame'
        time.sleep(0.001)
    # 换数据集要修改
    # im = Image.open('/home/hx/ORB_SLAM3/data/dataset-corridor5_512_16/mav0/cam0/data/' + fields[1] + '.png')
    # im = Image.open('/home/hx/ORB_SLAM3/data/dataset-room6_512_16/mav0/cam0/data/' + fields[1] + '.png')
    # im = Image.open('/home/hx/ORB_SLAM3/data/MH01/mav0/cam0/data/' + fields[1] + '.png')
    im = Image.open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/data/MH01/mav0/cam0/data/' + fields[1] + '.png')
    # im = Image.open('/home/hx/ORB_SLAM3/data/MH04/mav0/cam0/data/' + fields[1] + '.png')
    # im = Image.open('/home/hx/ORB_SLAM3/test_data/rgb/mav0/cam0/data/' + fields[1] + '.png')

    global imgname
    imgname = fields[1]
    # im = np.array(im.resize((MAZE_H, MAZE_W)))  # resize为输出图像尺寸大小，np.array将数据转化为矩阵
    # im = torch.from_numpy(np.array(im.resize((MAZE_H, MAZE_W)))).to(device)  # assuming you are using a GPU
    im = torch.from_numpy(np.array(im.resize((MAZE_H, MAZE_W))))
    errs = getError(fields, done)
    reward = 1 - errs
    return im, reward

def getError(fields, done):

    global ground, ground_temp, estimate_temp
    global estimate
    global true_traj
    global true_traj_count
    timeStep = fields[2]
    timeStep_temp = timeStep.replace(".", "")[:-1] + "0000"
    flag = False
    count = true_traj_count
    while count < len(true_traj):
        line_true = true_traj[count].strip()  # str.strip()就是把字符串(str)的头和尾的空格，以及位于头尾的\n \t之类给删掉。
        fields_true = line_true.split(' ')  # 拆分字符串。通过指定分隔符对字符串进行切片，并返回分割后的字符串列表（list）
        # 换数据集要修改
        # mh01:1403636580863550000   MH02:1403636859551660000
        # mh03=1403637133238310000,1403637133438310000   mh04=1403638129545090000  mh05=1403638522877820000
        # v101:1403715278762140000  1403715274312140000  1403715279012140000  V102:1403715529662140000
        # v103:1403715893884060000
        # v201:1413393217055760000   v202:1413393889205760000   v203:1413394887355760000    1413394887305760000
        if fields_true[0] == timeStep_temp:  # 如果估计轨迹的时间戳等于真实轨迹的时间戳
            if done and (timeStep_temp != "1403636580863550000"):
                true_traj_count = true_traj_count + 1
                ground = np.append(ground, [[float(fields_true[1]), float(fields_true[2]), float(fields_true[3])]],
                                   axis=0)
                estimate = np.append(estimate, [[float(fields[3]), float(fields[4]), float(fields[5])]], axis=0)
            else:
                count = count + 1
                ground[len(ground) - 1][0] = float(fields_true[1])
                ground[len(ground) - 1][1] = float(fields_true[2])
                ground[len(ground) - 1][2] = float(fields_true[3])
                estimate[len(estimate) - 1][0] = float(fields[3])
                estimate[len(estimate) - 1][1] = float(fields[4])
                estimate[len(estimate) - 1][2] = float(fields[5])
            flag = True
        count = count + 1
        if flag:
            break

    if len(ground) > 2:
        ground_temp = np.transpose(ground)
        estimate_temp = np.transpose(estimate)
        # Recover R and t
        ret_R, ret_t = rigid_transform_3D(estimate_temp, ground_temp)
        # Compare the recovered R and t with the original
        estimate_temp_val = (ret_R @ estimate_temp) + ret_t
        err = ground_temp - estimate_temp_val
        err = err * err  # 点乘
        err = np.sum(err)
        rmse = np.sqrt(err / len(ground))  # 均方根误差
    else:
        rmse = 0.50001
    # 换数据集要修改
    if len(ground) == 3637:  # mh01 = 3637  MH02=2995   MH03=2624   mh04=1965  mh05=1509
        # v101 = 2777    v102=1597   v103=1984   v201=2076   v202=2274  v203=1857
        # ground_3D = open('/home/hx/ORB_SLAM3/ground.txt', 'a', encoding='utf-8')
        # estimate_3D = open('/home/hx/ORB_SLAM3/estimate.txt', 'a', encoding='utf-8')
        #
        ground_3D = open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/ground.txt', 'a', encoding='utf-8')
        estimate_3D = open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/estimate.txt', 'a', encoding='utf-8')
        ground_temp_save = np.transpose(ground_temp).tolist()
        estimate_temp_save = np.transpose(estimate_temp).tolist()  # estimate_temp_val
        # f.write('%d %d %f %d %f\n' % (num_params, num_episodes, params[0], params[1], params[2]))
        for i in range(len(ground)):
            gr_str = str(ground_temp_save[i][0]) + ' ' + str(ground_temp_save[i][1]) + ' ' + str(
                ground_temp_save[i][2]) + '\n'
            es_str = str(estimate_temp_save[i][0]) + ' ' + str(estimate_temp_save[i][1]) + ' ' + str(
                estimate_temp_save[i][2]) + '\n'
            estimate_3D.write(es_str)
            ground_3D.write(gr_str)
        estimate_3D.close()
        ground_3D.close()
    return rmse

def run_maze(variant):
    # global num_params # 声明变量为全局变量
    # global num_frames
    # global num_episodes
    num_params = 0
    num_frames = 0
    num_episodes = 0
    step = 0
    t = 0
    device = variant['device']

    # torch.tensor(注意这里是小写)仅仅是python的函数, 函数原型是
    # torch.tensor(data, dtype=None, device=None, requires_grad=False)
    # 其中data可以是: list, tuple, NumPy, ndarray等其他类型, torch.tensor会从data中的数据部分做拷贝(
    # 而不是直接引用), 根据原始数据类型生成相应的torch.LongTensor torch.FloatTensor和torch.DoubleTensor

    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    ep_return = env_targets / scale
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    warmup_steps = variant['warmup_steps']
    epoch = variant['epoch']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    #LambdaLR也就可以理解成自定义规则去调整网络的学习率。从另一个角度理解，数学中的 λ \lambda λ一般是作为系数使用，
    # 因此这个学习率调度器的作用就是将初始学习率乘以人工规则所生成的系数 λ \lambda λ。
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )
    store = Store(state_dim=state_dim,
        act_dim=act_dim,
        max_length=20,
        max_ep_len=max_ep_len,
        scale = scale,
        mem_size=20000)
    #使用mh01训练
    for episode in range(3680):#MH01 = 3680  MH02=3040  MH03=2700  mh04=2033  mh05=2273
                                 #v101 = 2912  v102=1710  v103=2149  v201=2280  v202=2348  v203=1922
        print("=============================start==================================================")
        countt = 0
        pre_reward = 0     #预先将奖励值设为0
        control_reward_max_number = 0  #控制奖励值  防止奖励值
        while True:
            if countt > 1 or control_reward_max_number > 10:      #表示持续多少次不做优化
                done = True
                num_episodes += 1
            else:
                done = False
            # done = True
            observation, reward = get_observation(done=done) # 获得状态

            observation = observation.to(device=device, dtype=torch.float32)
            # observation = torch.from_numpy(observation).to(device=device, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            # RL choose action based on observation  # 基于观测选择动作
            action = model.get_action(
                (observation.to(dtype=torch.float32)),
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            actions[-1] = action
            # actions= action

            rewards[-1] = reward
            pred_return = target_return[0,-1] - (reward/scale)
            print("pred_return:",pred_return)
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (num_params+1)], dim=1)

            action1 = action.detach().cpu().numpy()
            action_max = np.argmax(abs(action1))
            a = (action_max).astype(np.int32)
            # observe = observation.flatten()
            # print('reward,pre_reward', reward, pre_reward)
            # RL take action and get next observation and reward
            control_reward_max_number += 1
            if reward > pre_reward:
                pre_reward = reward
                countt = 0
            else:
                countt += 1
            # observation_ = observation
            num_params = num_params + 1
            params = env.parameter_space[a]

            while True:
                succ = False
                try:
                    # f = open('/home/hx/ORB_SLAM3/read.txt', 'w')
                    # f_all = open('/home/hx/ORB_SLAM3/read_all.txt', 'a')
                    f = open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/read.txt', 'w')
                    f_all = open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/read_all.txt', 'a')

                    f.write('%d %d %f %d %f\n' % (num_params, num_episodes, params[0], params[1], params[2]))
                    f_all.write('%d %d %f %d %f\n' % (num_params, num_episodes, params[0], params[1], params[2]))
                    print('num_params:{}     countt:{}     num_episodes:{}'.format(num_params, countt, num_episodes))
                    succ = True
                finally:
                    if f:
                        f.close()
                    if f_all:
                        f_all.close()
                if succ:
                    break
                print('wait for SLAM processing ...')
                time.sleep(0.001)
            # 向內存回忆单元（s,a,r,s_）值num_params,countt,num_episodes
            observation = observation.cpu()
            action = action.cpu()
            reward = reward.cpu()
            # t = t.cpu()
            store.store_transition(observation, action, reward)
            del observation
            #torch.cuda.empty_cache()

            # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
            if (step % 3 == 0) and (step > 1500):   #step > 200

                trainer = SequenceTrainer(
                    model=model,
                    optimizer=optimizer,
                    batch_size=32,
                    # state_mean=state_mean,
                    # state_std=state_std,
                    get_batch=store.get_batch(32, 20),
                    scheduler=scheduler,
                    loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
                    epoch=epoch,
                )
                outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], print_logs=True)
                print(outputs)
            step += 1
            t += 1
            # break while loop when end of this episode
            if done:
                # num_episodes += 1
                break
        # trainer = SequenceTrainer(
        #     model=model,
        #     optimizer=optimizer,
        #     batch_size=32,
        #     # state_mean=state_mean,
        #     # state_std=state_std,
        #     get_batch=store.get_batch(32, 20),
        #     scheduler=scheduler,
        #     loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        #     epoch=epoch,
        # )
        # outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], print_logs=True)
        # print(outputs)
        print("episode", episode)
        if episode > 150:
            model.save_model(model,optimizer)
        print("*************************end*********************************")

    print('game over')
    # env.destroy()

def run_maze1(variant):
    num_params = 0
    num_frames = 0
    num_episodes = 0
    step = 0
    t = 0
    device = variant['device']
    actions = torch.zeros((0, act_dim),  dtype=torch.float32)
    rewards = torch.zeros(0,  dtype=torch.float32)

    # torch.tensor(注意这里是小写)仅仅是python的函数, 函数原型是
    # torch.tensor(data, dtype=None, device=None, requires_grad=False)
    # 其中data可以是: list, tuple, NumPy, ndarray等其他类型, torch.tensor会从data中的数据部分做拷贝(
    # 而不是直接引用), 根据原始数据类型生成相应的torch.LongTensor torch.FloatTensor和torch.DoubleTensor
    ep_return = env_targets/scale
    target_return = torch.tensor(ep_return, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0,  dtype=torch.long).reshape(1, 1)

    warmup_steps = variant['warmup_steps']
    epoch = variant['epoch']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    #LambdaLR也就可以理解成自定义规则去调整网络的学习率。从另一个角度理解，数学中的 λ \lambda λ一般是作为系数使用，
    # 因此这个学习率调度器的作用就是将初始学习率乘以人工规则所生成的系数 λ \lambda λ。
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )
    store = Store(state_dim=state_dim,
        act_dim=act_dim,
        max_length=20,
        max_ep_len=max_ep_len,
        scale = scale,
        mem_size=20000)
    #使用mh01训练

    for t in range(3680):#MH01 = 3680  MH02=3040  MH03=2700  mh04=2033  mh05=2273
                                 #v101 = 2912  v102=1710  v103=2149  v201=2280  v202=2348  v203=1922
        while True:

            done = True
            observation, reward = get_observation(done=done)
            # print('num_frames', num_frames)
            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim))], dim=0)
            observation = torch.tensor(observation, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)
            rewards = torch.cat([rewards, torch.zeros(1)])

            # target_return = target_return + rewards
            action = model.get_action(
                observation.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            actions[-1] = action
            action1 = action.detach().cpu().numpy()
            rewards[-1] = reward
            action_max = np.argmax(abs(action1))
            a = (action_max).astype(np.int32)

            params = env.parameter_space[a]
            num_params = num_params + 1
            num_episodes = num_episodes + 1
            while True:
                succ = False
                try:
                    # f = open('/home/hx/ORB_SLAM3/read.txt', 'w')
                    # f_all = open('/home/hx/ORB_SLAM3/read_all.txt', 'a')

                    f = open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/read.txt', 'w')
                    f_all = open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/read_all.txt', 'a')
                    f.write('%d %d %f %d %f\n' % (num_params, num_episodes, params[0], params[1], params[2]))
                    f_all.write('%d %d %f %d %f\n' % (num_params, num_episodes, params[0], params[1], params[2]))
                    print('num_params, num_episodes', num_params, num_episodes)
                    succ = True
                finally:
                    if f:
                        f.close()
                if succ:
                    break
                print('wait for SLAM processing ...')
                time.sleep(0.001)
            observation = observation.cpu()
            action = action.cpu()
            reward = reward.cpu()
            # t = t.cpu()
            store.store_transition(observation, action, reward)
            del observation
            torch.cuda.empty_cache()
            if t>1000 and t%8==0:
                trainer = SequenceTrainer(
                    model=model,
                    optimizer=optimizer,
                    batch_size=64,
                    # state_mean=state_mean,
                    # state_std=state_std,
                    get_batch=store.get_batch(64, 20),
                    scheduler=scheduler,
                    loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
                    epoch=epoch,
                )
                outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], print_logs=True)
                print(outputs)
            pred_return = target_return[0,-1] - (reward/scale)
            print("pred_return:",pred_return)
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), dtype=torch.long) * (num_params+1)], dim=1)

            if done:
                break

        print("episode", t)
        if t > 1900:
            model.save_model(model,optimizer)
        print("*************************end*********************************")

    print('game over')

def test(
        model,
        variant,
):
    mode = variant.get('mode', 'normal')

    num_eval_episodes = variant['num_eval_episodes']
    device = variant['device']

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                # 包装器“ with torch.no_grad（）”将所有require_grad标志临时设置为false。
                # 不希望PyTorch计算新定义的变量param的梯度（减少计算量），因为他只想更新它们的值。
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        scale=scale,
                        target_return=target_rew/scale,
                        mode=mode,
                        # state_mean=state_mean,
                        # state_std=state_std,
                        device=device,
                    )
                returns.append(ret)
                lengths.append(length)
        return fn

    warmup_steps = variant['warmup_steps']
    print(model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    print(model.parameters())
    #LambdaLR也就可以理解成自定义规则去调整网络的学习率。从另一个角度理解，数学中的 λ \lambda λ一般是作为系数使用，
    # 因此这个学习率调度器的作用就是将初始学习率乘以人工规则所生成的系数 λ \lambda λ。

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    trainer = Test(
        model=model,
        scheduler=scheduler,
        eval_fns=[eval_episodes(env_targets)],
    )

    # model.load_model(model, optimizer)
    # model.eval()

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        print(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse标准设置为正常，稀疏设置为延迟
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision-transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--num_eval_episodes', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=1)
    parser.add_argument('--num_steps_per_iter', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--epoch', type=int, default=30, help="训练次数")
    
    args = parser.parse_args()

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=20,
        max_ep_len=max_ep_len,
        hidden_size=64,
        n_layer=3,
        n_head=1,
        n_inner=4 * 64,
        activation_function='relu',
        # n_positions=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        mem_size=20000,
        is_train=False,
        scale=1000,
        # device='cuda'
        device='cpu'
    )
    model = model.to('cpu')
    # run_maze(variant=vars(args))
    test(model,variant=vars(args))
