import numpy as np
import torch
import time
from PIL import Image
from maze_env import MAZE_H, MAZE_W
from rigid_transform_3D import rigid_transform_3D
from decision_transformer import FeatureExtractor
feature_extractor = FeatureExtractor()
def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        scale=1000.,
        # state_mean=0.,
        # state_std=1.,
        device='cpu',
        target_return=None,
        mode='normal',
    ):
    # model.eval()作用等同于 self.train(False)
    # 简而言之，就是评估模式。而非训练模式。
    # 在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。
    model.eval()
    model.to(device=device)

    # torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
    # state_mean = torch.from_numpy(state_mean).to(device=device)
    # state_std = torch.from_numpy(state_std).to(device=device)

#=============================加
    global ground, ground_temp, estimate_temp
    global estimate
    # MH01
    # ground = np.array([[4.687579000000000384e+00, -1.786059000000000063e+00, 8.035400000000000320e-01]])
    # estimate = np.array([[-0.013096054, 0.148018956, 0.051851347]])
    # MH02
    # ground = np.array([[4.620760999999999896e+00, -1.836673999999999918e+00, 7.462400000000000144e-01]])
    # estimate = np.array([[0.079394445, -0.285317361, -0.085732698]])
    # MH03
    # ground = np.array([[4.637800000000000367e+00, -1.784734000000000043e+00, 5.946029999999999927e-01]])
    # estimate = np.array([[0.004397152, -0.022089170, -0.012676725]])
    # MH04
    # ground = np.array([[4.681549000000000404e+00, -1.751622999999999930e+00, 6.035420000000000229e-01]])
    # estimate = np.array([[-0.005734304, -0.027349519, -0.008012845]])
    # MH05
    # ground = np.array([[4.476938999999999780e+00, -1.649105999999999961e+00, 8.731020000000000447e-01]])
    # estimate = np.array([[-0.002883305, -0.031148085, -0.005559548]])

    # V101
    # ground = np.array([[9.528940000000000188e-01, 2.17603800000000013,

    # V102  5.153419999999999668e-01 1.996723000000000026e+00 9.710769999999999680e-01    -0.012740707 -0.042944785 -0.002622362
    # ground = np.array([[6.842449999999999921e-01, 2.080815999999999999e+00, 1.268488999999999978e+00]])
    # estimate = np.array([[-0.011741405, -0.014601516, 0.006989324]])

    # V103
    # ground = np.array([[8.840989999999999682e-01, 2.051204999999999945e+00, 1.011306999999999956e+00]])
    # estimate = np.array([[-0.011562885, -0.024513992, -0.016537875]])

    # V201   -1.067801999999999918e+00 4.953440000000000065e-01 1.372573999999999961e+00      0.000958465 -0.015630638 -0.006562878
    # ground = np.array([[-1.067801999999999918e+00, 4.953440000000000065e-01, 1.372573999999999961e+00]])
    # estimate = np.array([[0.000958465, -0.015630638, -0.006562878]])

    # V202
    # ground = np.array([[-1.004812999999999956e+00, 4.789249999999999896e-01, 1.331321999999999894e+00]])
    # estimate = np.array([[0.000083575, -0.006027541, -0.002623373]])

    # V203   -1.047128000000000059e+00 4.339850000000000096e-01 1.362441000000000013e+00
    # ground = np.array([[-1.047128000000000059e+00, 4.339850000000000096e-01, 1.362441000000000013e+00]])
    # estimate = np.array([[0.003970685, -0.011897060, -0.007381093]])
    #R1
    # ground = np.array([[8.657856723000000310e-01, -2.206935597999999943e-01, 1.264386139700000022e+00]])
    # estimate = np.array([[-0.001211587, -0.000130579, -0.000014503]])
    #R2
    # ground = np.array([[6.307559248999999868e-01, -3.432439313000000269e-01, 1.261092546700000039e+00]])
    # estimate = np.array([[-0.002903340, -0.000555362, -0.004095694]])
    #r3
    # ground = np.array([[1.241350252299999957e+00, -3.640493751000000167e-01, 1.238725601799999909e+00]])
    # estimate = np.array([[-0.000687140, -0.000069694, 0.000156301]])
    #r4
    # ground = np.array([[8.077123494000000292e-01, -2.664604503999999929e-01, 1.267281501400000066e+00]])
    # estimate = np.array([[-0.000770109, -0.000275534, 0.009585458]])
    #r5
    # ground = np.array([[6.307559248999999868e-01, -3.432439313000000269e-01, 1.261092546700000039e+00]])
    # estimate = np.array([[-0.002903340, -0.000555362, -0.004095694]])
    #r6
    # ground = np.array([[6.307559248999999868e-01, -3.432439313000000269e-01, 1.261092546700000039e+00]])
    # estimate = np.array([[-0.002903340, -0.000555362, -0.004095694]])
    #c1
    # ground = np.array([[6.918464968999999964e-01, -2.294512908000000129e-01, 1.289354945199999936e+00]])
    # estimate = np.array([[-0.019873118, -0.006652693, -0.001004581]])
    #c2
    # ground = np.array([[6.940819973999999837e-01, -4.841210786999999849e-01, 1.396389935400000004e+00]])
    # estimate = np.array([[0.004758288, -0.002774665, -0.004545398]])
    #c3
    # ground = np.array([[5.704181784999999838e-01, -5.124023764999999786e-01, 1.345135726299999890e+00]])
    # estimate = np.array([[-0.019326201, -0.030403614, 0.005721679]])
    #c4
    ground = np.array([[4.386356261000000090e-01, -2.578727344000000254e-01, 1.291830881799999942e+00]])
    estimate = np.array([[0.001723437, 0.000564033, -0.008476660]])
    #c5
    # ground = np.array([[4.640186210000000200e-01, -2.510083053000000120e-01, 1.341612185800000079e+00]])
    # estimate = np.array([[-0.008709456, 0.002992609, -0.004427414]])
    num_params = 0
    global true_traj
    true_traj = []
    true_traj_count = 0


    f_true = open('/home/hx/ORB_SLAM3/truth_deal/true_corridor4.txt', 'r')
    # f_true = open('/home/hx/ORB_SLAM3/truth_deal/true_room4.txt', 'r')
    # f_true = open('/home/hx/ORB_SLAM3/truth_deal/true_v101.txt', 'r')
    # f_true = open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/truth_deal/true_m1.txt', 'r')
    while True:
        line_true = f_true.readline()  # 该方法每次读出一行内容，返回一个字符串
        true_traj.append(line_true)
        if not line_true:
            break
    f_true.close()

    def get_observation(done=False):
        global num_frames
        while True:
            num = 0
            try:
                f = open('/home/hx/ORB_SLAM3/result.txt', 'r')
                # f = open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/result.txt', 'r')
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
                print('new frame:', num, num_frames)
                break
            time.sleep(0.001)

        n = int(fields[0])
        with open('/home/hx/ORB_SLAM3/Examples/Monocular/TUM_TimeStamps/dataset-corridor4_512.txt', 'r') as file:
        # with open('/home/hx/ORB_SLAM3/Examples/Monocular/TUM_TimeStamps/dataset-room5_512.txt', 'r') as file:
            lines = file.readlines()
            if n <= len(lines):  # 确保 n 不大于文件的总行数
                line_content = lines[n - 1].rstrip()  # 行号从1开始, 所以需要减1
            else:
                line_content = None  # 文件中没有第 n 行
        # 换数据集要修改
        im = Image.open('/home/hx/ORB_SLAM3/data/dataset-corridor4_512_16/mav0/cam1/data/' + line_content + '.png')
        # im = Image.open('/home/hx/ORB_SLAM3/data/dataset-room5_512_16/mav0/cam0/data/' + line_content + '.png')
        # im = Image.open('/home/hx/ORB_SLAM3/data/MH02/mav0/cam0/data/' + fields[1] + '.png')
        # im = Image.open('/home/hx/ORB_SLAM3/data/V101/mav0/cam0/data/' + fields[1] + '.png')
        # im = Image.open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/data/MH01/mav0/cam0/data/' + fields[1] + '.png')
        # im = Image.open('/home/hx/ORB_SLAM3/test_data/rgb/mav0/cam0/data/' + fields[1] + '.png')

        global imgname
        imgname = fields[1]
        print(fields[1])
        # im = np.array(im.resize((MAZE_H, MAZE_W)))  # resize为输出图像尺寸大小，np.array将数据转化为矩阵
        im = torch.from_numpy(np.array(im.resize((MAZE_H, MAZE_W))))
        errs = getError(fields, done)
        reward = 1 - errs
        return im, reward

    def getError(fields, done):
        global ground, ground_temp, estimate_temp
        global estimate
        global true_traj
        global true_traj_count
        true_traj_count = 0
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
            # v101:1403715279012140000  V102:1403715529662140000
            # v103:1403715893884060000
            # v201:1413393217055760000   v202:1413393889205760000   v203:1413394887355760000    1413394887305760000
            #Room1:1520530308199450000   room2:1520530733100100000   room3:1520530963500190000  room4:1520531126950600000   room5:1520531469950700000  room6:1520621016637030000
            #corridor1:1520531830201170000   corridor2:1520616232707170000   corridor3:1520616731208050000   corridor4:1520621178737000000   corridor5:1520622052587190000

            if fields_true[0] == timeStep_temp:  # 如果估计轨迹的时间戳等于真实轨迹的时间戳
                if done and (timeStep_temp != "1520621178737000000"):
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
        if len(ground) == 1052:  # mh01 = 3637  MH02=2995   MH03=2624   mh04=1965  mh05=1509
            # v101 = 2777    v102=1597   v103=1984   v201=2076   v202=2274  v203=1857
            #room1 = 2757    room2=2548  room3=2516  room4=2122  room5=2782  room6=2815
            #corridor1 = 1364   corridor2 = 1375   corridor3 = 1061   corridor4 = 1052   corridor5 = 795
            ground_3D = open('/home/hx/ORB_SLAM3/ground.txt', 'a', encoding='utf-8')
            estimate_3D = open('/home/hx/ORB_SLAM3/estimate.txt', 'a', encoding='utf-8')
            # ground_3D = open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/ground.txt', 'a', encoding='utf-8')
            # estimate_3D = open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/estimate.txt', 'a', encoding='utf-8')
            ground_temp_save = np.transpose(ground_temp).tolist()
            estimate_temp_save = np.transpose(estimate_temp).tolist()  # estimate_temp_val
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

    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    # torch.tensor(注意这里是小写)仅仅是python的函数, 函数原型是
    # torch.tensor(data, dtype=None, device=None, requires_grad=False)
    # 其中data可以是: list, tuple, NumPy, ndarray等其他类型, torch.tensor会从data中的数据部分做拷贝(
    # 而不是直接引用), 根据原始数据类型生成相应的torch.LongTensor torch.FloatTensor和torch.DoubleTensor
    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    global num_frames
    num_frames = 0
    num_episodes = 1
    episode_return, episode_length = 0, 0
    for t in range(1908):#MH01 = 3680  MH02=3040  MH03=2700  mh04=2033  mh05=2273
                         # v101 = 2912  v102=1710  v103=2149  v201=2280  v202=2348  v203=1922
                         # room1 = 2727 room2=2848  room3=2806  room4=2172   room5=2799  room6=2595
                         # corridor1 = 5971  corridor2 = 6727   corridor3 = 5665   corridor4 = 1908/1300   corridor5 = 5899
        while True:

                done = True
                states, reward = get_observation(done=done)
                # states = states.reshape(1, -1, state_dim).float()
                # feature = feature_extractor(states)
                # feature = torch.tensor(feature, device=device, dtype=torch.float32)
                feature = torch.tensor(states, device=device, dtype=torch.float32)
                actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
                reward = torch.tensor(reward, device=device, dtype=torch.float32)
                rewards = torch.cat([rewards, torch.zeros(1, device=device)])
                # target_return = target_return + rewards
                action = model.get_action(
                    # (states.to(dtype=torch.float32) - state_mean) / state_std,
                    feature.to(dtype=torch.float32),
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
                actions[-1] = action
                action = action.detach().cpu().numpy()
                rewards[-1] = reward
                action_max = np.argmax(abs(action))
                a = (action_max).astype(np.int32)
                params = env.parameter_space[a]
                num_params = num_params + 1
                num_episodes = num_episodes + 1
                while True:
                    succ = False
                    try:
                        f = open('/home/hx/ORB_SLAM3/read.txt', 'w')
                        f_all = open('/home/hx/ORB_SLAM3/read_all.txt', 'a')
                        # f = open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/read.txt', 'w')
                        # f_all = open('/remote-home/cs_ai_hch/hx/ORB_SLAM3-master/read_all.txt', 'a')
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


                pred_return = target_return[0,-1] - (reward/scale)
                print("pred_return:",pred_return)
                # target_return = target_return - pred_return.reshape(1, 1)
                target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
                timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (num_params+1)], dim=1)

                episode_return += reward
                episode_length += 1

                # if done:
                #     break

    return episode_return, episode_length

