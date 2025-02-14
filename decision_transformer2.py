import numpy as np
import torch
import torch.nn as nn

import transformers

from model import TrajectoryModel
from trajectory_gpt2 import GPT2Model
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
            max_ep_len=6000,
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

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        # 注意：这个GPT2Model和默认的Huggingface版本之间的唯一区别是删除了位置嵌入（因为我们将自己添加这些嵌入）
        self.transformer = GPT2Model(config)
        self.embed_timestep = torch.nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)#从1扩展到hidden_size的维度
        #==========================
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper注意：我们不会预测论文的状态或回报
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)



    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        # embed each modality with a different head用不同的头部嵌入每个模态
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings 时间嵌入的处理方式与位置嵌入类似
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well为了使注意力掩码适合堆叠的输入，还必须将其堆叠
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model我们将输入嵌入（而不是NLP中的单词索引）输入到模型中
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:, 2])  # predict next return given state and action预测给定状态和动作的下一次返回
        state_preds = self.predict_state(x[:, 2])  # predict next state given state and action预测给定状态和动作的下一次状态
        action_preds = self.predict_action(x[:, 1])  # predict next action given state预测给定状态下的下一个动作

        return state_preds, action_preds, return_preds
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
        model_name = 'model_MH01-2_sev.pt'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, os.path.join(model_save_path, model_name))

        # 加载模型函数
    def load_model(self, model, optimizer):
        last_model_path = '../DT_model/'
        model_name = 'model_MH01-2_sev.pt'
        checkpoint = torch.load(os.path.join(last_model_path, model_name))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # model.eval()
