import numpy as np
import torch
import torch.nn as nn

import transformers

from model import TrajectoryModel
from trajectory_gpt2 import GPT2Model, Autodis,CNN,VectorPatchEmb,SABlock,Block1,GPTConfig
import os
class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            is_train,
            mem_size,
            scale,
            device,
            max_length=None,
            max_ep_len=15000,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.is_train=is_train
        self.hidden_size = hidden_size
        self.mem_size = mem_size
        self.max_ep_len = max_ep_len
        self.scale = scale
        self.device = device
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # 初始化第二个 Transformer 模块
        transformer1 = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        # 注意：这个GPT2Model和默认的Huggingface版本之间的唯一区别是删除了位置嵌入（因为我们将自己添加这些嵌入）
        self.transformer = GPT2Model(config)
        self.transformer1 = GPT2Model(transformer1)
        self.embed_timestep = torch.nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)#从1扩展到hidden_size的维度
#======
        self.cnn = CNN(self.is_train)
        bucket_number = 100
        self.ret_emb = Autodis(config, bucket_number)
        self.states = torch.empty((self.mem_size, self.state_dim, self.state_dim),dtype=torch.uint8)
        self.actions = torch.empty(self.mem_size, self.act_dim,dtype=torch.int32)
        self.rewards = torch.empty((self.mem_size), dtype=torch.int32)
        self.states_ = torch.empty((self.mem_size, self.state_dim, self.state_dim),dtype=torch.uint8)
        self.count = 0
        self.current = 0

#=============

        mconf = GPTConfig(vocab_size=1, block_size=64,
                          n_layer=3, n_head=1, n_embd=hidden_size, max_timestep=60)
        self.mconf = mconf


        # input embedding stem
        self.tok_emb = nn.Embedding(mconf.vocab_size, mconf.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, mconf.block_size, mconf.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, mconf.block_size , mconf.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, mconf.max_timestep, mconf.n_embd))
        self.drop = nn.Dropout(mconf.embd_pdrop)

        # transformer
        self.blocks1 = nn.Sequential(*[Block1(mconf) for _ in range(mconf.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(mconf.n_embd)
        self.head = nn.Linear(mconf.n_embd, mconf.vocab_size, bias=False)

        self.block_size = mconf.block_size
        self.apply(self._init_weights)


        self.state_encoder = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                 nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                 nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                 nn.Flatten(),nn.Tanh())

        # self.ret_emb = nn.Sequential(nn.Linear(1, mconf.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Embedding(mconf.vocab_size, mconf.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

        #=================================

        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)#从state_dim扩展到hidden_size的维度
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)#从act_dim扩展到hidden_size的维度
        self.embed_ln = nn.LayerNorm(hidden_size)#在hidden_size进行norm
        self.embed_ln = nn.LayerNorm(hidden_size)#在hidden_size进行norm
        # note: we don't predict states or returns for the paper注意：我们不会预测论文的状态或回报
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)#预言状态层  hidden_size-》state_dim
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        # self.predict_return = nn.Sequential(
        #     *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        # )
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        self.i = 0
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.act_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.act_dim)
        # )
        # self.predict_action = nn.Sequential(
        #     *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else [])),
        #     self.mlp
        # )
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        state_embeddings1 = self.state_encoder(states.type(torch.float32).contiguous()).view(1, 20, 64)  # (batch * block_size, n_embd)
        print("state_embeddings1", torch.cuda.max_memory_allocated() / 1024 ** 2)
        # state_embeddings1 = self.embed_state(states)
        state_embeddings1 = state_embeddings1[:, :, :64]
        # print("state_embeddings1", torch.cuda.max_memory_allocated() / 1024 ** 2)
        # state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1],self.mconf.n_embd)  # (batch, block_size, n_embd)

        if actions is not None:
            rtg_embeddings = self.ret_emb(returns_to_go.type(torch.float32))
            # print("rtg_embeddings", torch.cuda.max_memory_allocated() / 1024 ** 2)
            action_embeddings1 = self.embed_action(actions)
            # print("action_embeddings1", torch.cuda.max_memory_allocated() / 1024 ** 2)
            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 3 - int(rewards is None), self.mconf.n_embd), dtype=torch.float32,
                device=state_embeddings1.device)
            # print("token_embeddings", torch.cuda.max_memory_allocated() / 1024 ** 2)
            token_embeddings[:, ::3, :] = rtg_embeddings
            # print("token_embeddings[:, ::3, :]", torch.cuda.max_memory_allocated() / 1024 ** 2)
            token_embeddings[:, 1::3, :] = state_embeddings1
            # print("token_embeddings[:, 1::3, :]", torch.cuda.max_memory_allocated() / 1024 ** 2)
            token_embeddings[:, 2::3, :] = action_embeddings1[:, -states.shape[1] + int(rewards is None):, :]
            # print("token_embeddings[:, 2::3, :]", torch.cuda.max_memory_allocated() / 1024 ** 2)
            state_embeddings1 = state_embeddings1.cpu()
        # elif actions is None:  # only happens at very first timestep of evaluation
        #     rtg_embeddings = self.ret_emb(returns_to_go.type(torch.float32))
        #
        #     token_embeddings = torch.zeros((states.shape[0], states.shape[1] * 2, self.config.n_embd),
        #                                    dtype=torch.float32, device=state_embeddings1.device)
        #     token_embeddings[:, ::2, :] = rtg_embeddings  # really just [:,0,:]
        #     token_embeddings[:, 1::2, :] = state_embeddings1  # really just [:,1,:]
        # all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size,dim=0)  # batch_size, traj_length, n_embd
        timesteps1 = timesteps.repeat(1, 3)
        # print("timesteps1", torch.cuda.max_memory_allocated() / 1024 ** 2)
        time_embeddings1 = self.embed_timestep(timesteps1)
        # print("time_embeddings1", torch.cuda.max_memory_allocated() / 1024 ** 2)
        position_embeddings = time_embeddings1 + self.pos_emb[:, :token_embeddings.shape[1],:]
        # print("position_embeddings", torch.cuda.max_memory_allocated() / 1024 ** 2)

        y = self.drop(token_embeddings + position_embeddings)
        # print("y", torch.cuda.max_memory_allocated() / 1024 ** 2)
        y = self.blocks1(y)
        # print("y", torch.cuda.max_memory_allocated() / 1024 ** 2)
        y = self.blocks1(y)
        # print("y", torch.cuda.max_memory_allocated() / 1024 ** 2)
        y = self.ln_f(y)
        # print("y", torch.cuda.max_memory_allocated() / 1024 ** 2)
        y = y.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        return_preds1 = self.predict_return(y[:,2])  # predict next return given state and action预测给定状态和动作的下一次返回
        state_preds1 = self.predict_state(y[:,2])    # predict next state given state and action预测给定状态和动作的下一次状态
        action_preds1 = self.predict_action(y[:,1])  # predict next action given state预测给定状态下的下一个动作

        del states, actions, returns_to_go, timesteps, attention_mask
        del state_embeddings1
        i = self.i+1
        if i > 1000 and i % 100 == 0:
            torch.cuda.empty_cache()

        return state_preds1, action_preds1, return_preds1

# #==============
#         local_tokens, global_state_tokens, temporal_emb = self.token_emb(states, actions, rewards=rewards)
#         local_tokens = self.local_pos_drop(local_tokens)
#         global_state_tokens = self.global_pos_drop(global_state_tokens)
#         for i, blk in enumerate(self.blocks):
#             if i == 0:
#                 local_tokens, global_state_tokens, local_att, global_att = blk(local_tokens, global_state_tokens,
#                                                                                temporal_emb)
#             else:
#                 local_tokens, global_state_tokens, local_att, global_att = blk(local_tokens, global_state_tokens,
#                                                                                temporal_emb)
#
#         y = self.ln_head(global_state_tokens)
#===================

        # # Apply the CNN layers to the input 'states'
        # state = self.cnn(states)
        # if attention_mask is None:
        #     # attention mask for GPT: 1 if can be attended to, 0 if not GPT的注意掩码：如果可以关注，则为1，如果不可以关注，为0
        #     attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        #
        # # embed each modality with a different head用不同的头部嵌入每个模态
        # state_embeddings = self.embed_state(state)  #要等于（batch size,k,state）
        # action_embeddings = self.embed_action(actions)
        # returns_embeddings = self.ret_emb(returns_to_go)
        # time_embeddings = self.embed_timestep(timesteps)
        #
        # # time embeddings are treated similar to positional embeddings 时间嵌入的处理方式与位置嵌入类似
        # state_embeddings = state_embeddings + time_embeddings
        # action_embeddings = action_embeddings + time_embeddings
        # returns_embeddings = returns_embeddings + time_embeddings
        #
        # # 将两个tensor按照交叉组合的方式拼接起来
        # # result_tensor = torch.cat((y.unsqueeze(1), state_embeddings.unsqueeze(1)), dim=1)
        # #
        # # # 将结果tensor展平成一维
        # # input_states = result_tensor.view(-1).reshape(1,40,64)
        # # input_states = input_states[:,-self.max_length:]
        #
        # # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # # which works nice in an autoregressive sense since states predict actions
        # # 这使得序列看起来像（R_1，s_1，a_1，R_2，s_2，a_2，…），它在自回归意义上工作得很好，因为状态预测动作
        # #torch.stack()操作是对张量进行拼接操作,torch.stack()将对张量维度进行扩张
        # # stacked_inputs = torch.stack(
        # #     (returns_embeddings, state_embeddings, action_embeddings), dim=1
        # # ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        # stacked_inputs = torch.stack(
        #     (returns_embeddings, state_embeddings, action_embeddings), dim=1
        # ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        # stacked_inputs = self.embed_ln(stacked_inputs)
        #
        # # to make the attention mask fit the stacked inputs, have to stack it as well为了使注意力掩码适合堆叠的输入，还必须将其堆叠
        # stacked_attention_mask = torch.stack(
        #     (attention_mask, attention_mask, attention_mask), dim=1
        # ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        # # we feed in the input embeddings (not word indices as in NLP) to the model我们将输入嵌入（而不是NLP中的单词索引）输入到模型中
        # transformer_outputs = self.transformer(
        #     inputs_embeds=stacked_inputs,
        #     attention_mask=stacked_attention_mask,
        # )
        # x = transformer_outputs['last_hidden_state']
        #
        # # reshape x so that the second dimension corresponds to the original
        # # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # #重塑x，使第二个维度对应于原始返回（0）、状态（1）或动作（2）；即x[：，1，t]是s_t的令牌
        # # x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        #
        # #==
        # x = self.embed_ln(x)
        # transformer_outputs1 = self.transformer1(
        #     inputs_embeds=x,
        #     attention_mask=stacked_attention_mask,
        # )
        # x1 = transformer_outputs1['last_hidden_state']
        # x1 = x1.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        # #
        # # #=====
        # # # x2 = self.drop(stacked_inputs + stacked_attention_mask)
        # # x2 = self.blocks(x)
        # # x2 = self.ln_f(x2)
        # # x2 = x2.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        #
        # # get predictions
        # return_preds = self.predict_return(x1[:,2])  # predict next return given state and action预测给定状态和动作的下一次返回
        # state_preds = self.predict_state(x1[:,2])    # predict next state given state and action预测给定状态和动作的下一次状态
        # action_preds = self.predict_action(x1[:,1])  # predict next action given state预测给定状态下的下一个动作
        # del states, actions, returns_to_go, timesteps, attention_mask
        # del state_embeddings
        # del stacked_inputs
        # i = self.i+1
        # if i > 1000 and i % 100 == 0:
        #     torch.cuda.empty_cache()
        #
        # return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model我们不在乎这个模型过去的奖励

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = (returns_to_go).reshape(1, -1, 1)
        rewards = (rewards).reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length将所有令牌填充到序列长度
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            # print("attention_mask", torch.cuda.max_memory_allocated() / 1024 ** 2)
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            # print("attention_mask", torch.cuda.max_memory_allocated() / 1024 ** 2)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            # print("states", torch.cuda.max_memory_allocated() / 1024 ** 2)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions], dim=1).to(dtype=torch.float32)
            # print("actions", torch.cuda.max_memory_allocated() / 1024 ** 2)
            # rewards = torch.cat(
            #     [torch.zeros((rewards.shape[0], self.max_length-rewards.shape[1], 1),device=rewards.device), rewards],
            #     dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            # print("returns_to_go", torch.cuda.max_memory_allocated() / 1024 ** 2)

            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1).to(dtype=torch.long)
            # print("timesteps", torch.cuda.max_memory_allocated() / 1024 ** 2)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)
        print("++++++++++++++++", torch.cuda.max_memory_allocated() / 1024 ** 2)
        m=action_preds[0,-1]
        return action_preds[0,-1]

        # 保存模型函数

    def save_model(self, model, optimizer):
        model_save_path = '../DT_model/'
        # model_name = 'model_MH01-c2gpu_sev.pt'
        # model_name = 'model_MH01-4gpu_sev.pt'
        model_name = 'model_V101-gpu_sev.pt'
        # model_name = 'model_MH01-3_sev.pt'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, os.path.join(model_save_path, model_name))

        # 加载模型函数
    def load_model(self, model, optimizer):
        last_model_path = '../DT_model/'
        # model_name = 'model_MH01-c2gpu_sev.pt'
        model_name = 'model_MH01-4gpu_sev.pt'
        # model_name = 'model_V101-gpu_sev.pt'
        # model_name = 'model_MH01-3_sev.pt'
        checkpoint = torch.load(os.path.join(last_model_path, model_name))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # model.eval()

    def get_mean_std(self):
        states1 = np.concatenate(self.states, axis=0)
        state_mean, state_std = np.mean(states1, axis=0), np.std(states1, axis=0) + 1e-6
        print(state_mean, state_std)

        return state_mean, state_std

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.tanh(x)
        x = x.view(1, 64, 64)
        x = x[:, -20:]
        return x