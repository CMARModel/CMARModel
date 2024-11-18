import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.nn.init import normal_, xavier_normal_, xavier_uniform_
from models.layers import MLPLayers
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from itertools import product
import random
import pandas as pd
from models.layers import  ActorLayer, CriticLayer
import math
from models  import QNetwork
from trainer import *
torch.set_printoptions(threshold=np.inf)
torch.autograd.set_detect_anomaly(True)


class LoadFile(object):
    def __init__(self, sat_api_path, sat_api_cost_path, api_path, sat_count):
        self.sat_api_path = sat_api_path
        self.sat_api_cost_path = sat_api_cost_path
        self.api_path = api_path
        self.sat_count = sat_count
        self.load()

    def pad_sequence2fixlen(self, data, length):
        data = data + [torch.tensor([0 for i in range(length)])]
        data_pad = pad_sequence(data, batch_first=True, padding_value=0)
        return data_pad[:-1, :]

    def load(self):
        # 真实卫星和api的编号都从1开始
        # 处理sat_api,sat_api_cost,api文件，对应api在satellite的部署情况、能源消耗情况和api的属性
        # 增加id=0的卫星和api，预处理中横坐标增加id=0的api,纵坐标增加id=0的satellite
        # 但是id=0的api不在任何一个卫星上面，id=0的卫星上没有一个api
        sa_api_data = pd.read_csv(self.sat_api_path, names=[i for i in range(1, self.sat_count + 1)])
        self.api_sat = []
        # 增加一列, 增加id=0的卫星
        col = [0 for i in range(len(sa_api_data))]
        sa_api_data.insert(0, 0, col, allow_duplicates=False)
        # 增加一行, 增加id=0的api
        id_0_df = pd.DataFrame([[0 for i in range(sa_api_data.shape[1])]],
                               columns=[i for i in range(0, self.sat_count + 1)])
        sa_api_data = id_0_df.append(sa_api_data, ignore_index=True)
        self.api_sat_count = []
        for i in range(len(sa_api_data)):
            temp = np.array(sa_api_data.iloc[i, :].tolist())
            self.api_sat.append(torch.tensor(np.where(temp == 1)[0].tolist()))
            self.api_sat_count.append(len(np.where(temp == 1)[0].tolist()))
        # api_sat [api_all_count, sat_all_count] 每个api部署的卫星id
        # self.api_sat = pad_sequence(self.api_sat,  padding_value = 0, batch_first=True).cuda()
        self.api_sat = self.pad_sequence2fixlen(self.api_sat, self.sat_count + 1).cuda()
        # api_sat_count只统计了真实卫星的数量
        self.api_sat_count = torch.tensor(self.api_sat_count).cuda()

        # 记录调用satellite上api的能耗
        # 增加id=0的卫星和api
        self.api_id_money = {}
        cost_data = pd.read_csv(self.sat_api_cost_path, names=[i for i in range(1, self.sat_count + 1)])
        # 增加一列, 增加id=0的卫星
        col = [0 for i in range(len(cost_data))]
        cost_data.insert(0, 0, col, allow_duplicates=False)
        # 增加一行, 增加id=0的api
        id_0_df = pd.DataFrame([[0 for i in range(sa_api_data.shape[1])]],
                               columns=[i for i in range(0, self.sat_count + 1)])
        cost_data = id_0_df.append(cost_data, ignore_index=True)
        # cost_info [api_all_count, sat_all_count] 对应位置记录cost
        self.cost_info = torch.tensor(cost_data.values).cuda().float()

        # 每个api一个tensor，按顺序列举satellite的cost
        self.api_sat_cost = []
        self.api_sat_flat_cost = []
        # 认为0号卫星部署所有的api
        for i in range(len(cost_data)):
            temp = np.array(cost_data.iloc[i, :].tolist())
            self.api_sat_cost.append(torch.tensor(temp[np.where(temp > 0)[0].tolist()]))
        # 记录api部署satellite的cost, 与self.api_sat相对应，剩余位置用0 padding
        self.api_sat_cost = self.pad_sequence2fixlen(self.api_sat_cost, self.sat_count + 1).cuda().float()

        # 处理api_info文件，文件中已经包含编号为0的api
        # api 记录api文件信息
        # 3个特征 [api+1, 3]
        # CPU [4.5,70]
        # Memory [0.1,60]
        # Delay [5,7]
        api_data = pd.read_csv(self.api_path, names=["CPU", "Memory", "Delay"])
        # self.api_info: [api_all_count, 3]
        self.api_info = torch.tensor(api_data.values).cuda().float()


class CMAR(object):

    def __init__(self, config, state_dim):
        self.dataset = config.model_param['dataset']
        self.embedding_size = config.model_param['embedding_size']
        self.api_comb_max_length = config.model_param['api_comb_max_length']
        self.batch_size = config.train_batch_size
        self.gamma = 0.98
        self.K_repeats = 5
        self.eps_clip = 0.2
        self.epsilon = 1
        self.eps_dec = 5e-4
        self.eps_min = 0.05
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        # self.q_eval = QNetwork(weight_decay=self.weight_decay, learning_rate=self.learning_rate, **config.model_param).cuda()
        # self.q_target = QNetwork(weight_decay=self.weight_decay, learning_rate=self.learning_rate, **config.model_param).cuda()
        self.q_eval = QNetwork(state_dim=state_dim, weight_decay=self.weight_decay, learning_rate=self.learning_rate, **config.model_param).cuda()
        self.q_target = QNetwork(state_dim=state_dim, weight_decay=self.weight_decay, learning_rate=self.learning_rate, **config.model_param).cuda()
        self.set_hypers()
        # self.read_info()
        state_dim = (self.sat_count+1) * 7 + config.model_param["api_fea_count"] * 2
        self.memory = ReplayBuffer(config.buffer_size, config.train_batch_size, state_dim=self.sat_count+1, action_dim=state_dim)
        self.tau = 0.005
        self.update_network_parameters(tau=1.0)



    def set_hypers(self):
        # 此处记录api和satellite的真实数量
        self.sat_count = 30
        self.api_count = 200
        # xx_all_count 数量均包含api/satellite编号为0的虚拟api/sat
        # sat_all_count = sat_count+1
        # api_all_count = api_count+1
        # 记录satellite的特征数量
        self.sat_fea_count = 5
        # 记录api文件里记录的特征数量，此外在api_sat文件里面还包含api的cost属性
        self.api_fea_count = 3
        self.MseLoss = nn.MSELoss()

    def read_info(self):
        self.sat_api_path = 'dataset/%s/satellite_api.csv' % (self.dataset)
        self.sat_api_cost_path = 'dataset/%s/satellite_api_cost.csv' % (self.dataset)
        self.api_path = 'dataset/%s/api_info.csv' % (self.dataset)
        self.sat_count = 30
        fileloader = LoadFile(self.sat_api_path, self.sat_api_cost_path, self.api_path, self.sat_count)
        # api_sat [api_all_count, sat_all_count] 每个api部署的卫星id,用0进行padding
        self.api_sat = fileloader.api_sat
        # api_sat_count只统计了真实卫星的数量 [api_all_count, 1]
        self.api_sat_count = fileloader.api_sat_count
        # 横纵坐标对应位置的cost
        self.cost_info = fileloader.cost_info
        # api部署的卫星的cost,与self.api_sat一一对应
        self.api_sat_cost = fileloader.api_sat_cost
        # [api_all_count, 3]
        self.api_info = fileloader.api_info

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min


    def sample_action(self, state):
        actions = self.q_eval(state)
        action = torch.argmax(actions).item()

        if np.random.random() < self.epsilon:
            action = np.random.choice(np.arange(1, self.sat_count+1))

        return action

    def update(self):
        self.q_eval.train()
        state, action, reward, next_state, over = self.memory.sample_buffer()
        # import ipdb
        # ipdb.set_trace()
        state = torch.stack(state, dim=0).cuda()
        reward = torch.stack(reward, dim=0).cuda().reshape(-1,1)
        next_state = torch.stack(next_state, dim=0).cuda()
        over = torch.tensor(over).cuda().reshape(-1,1)
        # state = torch.tensor(state, dtype=torch.float32).cuda()
        # reward = torch.tensor(reward, dtype=torch.float32).cuda()
        # next_state = torch.tensor(next_state, dtype=torch.float32).cuda()
        # over = torch.tensor(over, dtype=torch.float32).cuda()

        with torch.no_grad():
            q_next = self.q_eval(next_state)
            next_action = torch.argmax(q_next,dim=-1)
            q_next = self.q_target(next_state)
            q_next = q_next*(1-over)
            batch_index =  np.arange(self.batch_size)
            target = reward + self.gamma * q_next[batch_index, next_action].reshape(-1,1)
        q = self.q_eval(state)[batch_index, action].reshape(-1,1)

        loss = self.MseLoss(q, target.detach())
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.update_network_parameters()
        self.decrement_epsilon()



