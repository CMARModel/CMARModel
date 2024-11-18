from torch.distributions import Categorical
import os
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import random
import pandas as pd
from collections import namedtuple
from sklearn import preprocessing
from utils import *


get_sample_data_collect = namedtuple('sample_data', ['states', 'next_states', 'rewards',
               'actions', 'overs', 'traj_flags', 'traj_count', 'batch_count'])


def get_index_order(old_probs):
    """

    Args:
        old_probs: [B, sat_all_count]

    Returns:
        index_list:[B, sat_all_count]
    """
    index_list = []
    length = old_probs.shape[1]
    mask_prob = old_probs
    mask = torch.full([mask_prob.shape[0], mask_prob.shape[1]], -np.inf).cuda()
    for i in range(length):
        mask_index = torch.argmax(mask_prob, dim=-1)
        index_list.append(mask_index)  # [B]
        mask_prob = mask_prob.scatter(1, mask_index.reshape(-1, 1), mask)
    index_tensor = torch.stack(index_list, dim=-1)
    return index_tensor

class Input(object):

    def __init__(self, **kwgs):
        for key, value in kwgs.items():
            value = torch.stack(value, dim=0)
            setattr(self, key, value)

    def cuda(self):
        for key in self.__dict__:
            setattr(self, key, getattr(self, key).cuda())
        return self



class Env(object):
    def __init__(self, policy):
        self.policy = policy
        self.embedding_size = self.policy.embedding_size
        # v0版本,用于辅助计算得到当前action
        self.count = 0  #下一次要为第i个api选取卫星
        self.action_space = self.policy.sat_count+1

    def get_state_concat(self, api):
        # 已废弃的版本
        # satellite属性现状+api是否被选择+api已选择的卫星编号(未选择则默认为0)+api在卫星上的cost+api的其他属性
        # [B, sat_all_count*5]
        # sat_info = self.sat_fea.reshape(self.batch_size, -1)
        #
        # api_sat_cost = self.api_sat_cost.repeat(self.batch_size, 1, 1).reshape(self.batch_size, -1)  # [B, api_all_count*sat_all_count]
        #
        # # [B, api_all_count*3]
        # api_fea = self.api_fea.repeat(self.batch_size, 1, 1).reshape(self.batch_size, -1)
        #
        # all_info = torch.cat([sat_info, self.choose, api_sat_cost, api_fea], axis=-1)

        # v0.0 测试版本，因为mask掉不可行的卫星，模拟贪心的逻辑，只展示当前api部署到不同satellite的cost
        # 此版本已测试,算是成功
        # obs:当前api所有sat的cost + sat的属性
        # api_cost = self.api_sat_cost[api]
        # api_cost = api_cost.reshape(-1,1)
        # # sat_fea = self.sat_fea.reshape(self.policy.sat_count+1, -1)
        # zscore = preprocessing.StandardScaler()
        # # sat_fea_norm = torch.zeros_like(sat_fea)
        # # sasct_fea_norm[1:, :] = torch.FloatTensor(zscore.fit_transform(sat_fea[1:, :].cpu()))
        # # sat_fea_norm = sat_fea_norm.reshape(-1)
        # # # all_info = torch.cat([api_cost,  sat_fea_norm], dim=-1)
        # # all_info = sat_fea_norm.reshape(-1)
        # all_info_norm = torch.zeros_like(api_cost)
        # all_info_norm[1:, :] = torch.FloatTensor(zscore.fit_transform(api_cost.reshape(-1,1)[1:,:].cpu()))
        # all_info = all_info_norm.reshape(-1)

        #v0.1
        """
        # obs:
        sat的属性[sat_all_count, sat_fea_count]
        当前api所有sat的cost[sat_all_count, 1]
        当前api的fea [api_fea_count]
        未选择的api的cost sum pooling [sat_all_count,1]
        未选择的api的fea sum pooling [api_fea_count]
        """
        # 当前api
        api_cost = self.api_sat_cost[api]
        api_fea = self.api_fea[api]
        # sat属性
        sat_fea = self.sat_fea.reshape(self.policy.sat_count+1, -1)
        #future api的sum pooling
        fut_api = self.api_comb[self.count+1:self.input.api_count.item()]

        fut_api_cost = self.api_sat_cost[fut_api,:].sum(dim=0)
        fut_api_fea = self.api_fea[fut_api,:].sum(dim=0)

        # all_info = api_cost #[sat_all_count]
        all_info = torch.cat([api_cost.reshape(-1, 1), sat_fea, fut_api_cost.reshape(-1, 1)], dim=-1)
        zscore = preprocessing.StandardScaler()
        all_info_norm = torch.zeros_like(all_info)
        all_info_norm[1:, :] = torch.FloatTensor(zscore.fit_transform(all_info[1:, :].cpu()))
        all_info = all_info_norm.reshape(-1)
        all_info = torch.cat([all_info,  api_fea, fut_api_fea], dim=0)

        return all_info

    def reset(self, input):
        self.input = input
        self.api_comb = self.input.api_comb[0]  # [api_comb_max_length]
        self.api_comb_max_length = self.api_comb.shape
        self.api_all_count, self.sat_all_count = self.policy.cost_info.shape
        # [sat_all_count, 5]
        self.sat_fea = self.input.sat_fea[0]
        # [B, api_comb_max_length, sat_all_count]
        # self.api_sat_cost = torch.gather(input=self.model.cost_info.repeat(self.batch_size,1,1), dim=1, index=self.api_comb.repeat(1, 1, self.sat_all_count))
        # [api_all_count, sat_all_count]
        self.api_sat_cost = self.policy.cost_info
        # [api_all_count, 3]
        self.api_fea = self.policy.api_info
        self.count = 0

        return self.get_state_concat(self.api_comb[0])

    def parse_action(self, action):
        # action id进行解析，返回选择的api和卫星
        return self.api_comb[self.count], action

    def step(self, action):
        '''
            返回next_state, reward, over
            并且对class中的原始数据进行更新，对以下四项进行更新
            self.sat_fea sat属性现状
            self.count 当前对哪一个api选择卫星
        '''
        # reward: -cost+成功与否(成功则为0，失败则为-10)
        # 因为是一个batch一起进行计算，所以如果失败了也继续采样，但是不会append到列表（利用over进行筛选）
        # 具体来说，就是第一个失败的action会被记录，而后续的失败action不会被记录，通过记录last_overs得到
        '''
            先取出action的cost和对应api属性消耗，记录reward
            从sat_fea和api_sat_cost里面减去，更新sat_fea
            再判断新的state是否合法，从而更新over,再据此对state进行mask
        '''
        api, sat = self.parse_action(action)


        # 先计算卫星api对应的reward/cost
        cost = self.api_sat_cost[api][sat]
        reward = -cost #[B,1]
        # if sat == 1:
        #     reward = -0.001
        if sat >=20:
            reward = torch.tensor(-0.0001).cuda()

        # 计算api消耗的属性
        # [api_all_count, 3]
        api_fea = self.api_fea[api] #[3]


        # 对现有state的sat的CPU、Memory、Energy进行更新
        # self.sat_fea [sat_all_count, 5]
        # 先更新energy
        self.sat_fea[sat, SAT_ENG] = self.sat_fea[sat, SAT_ENG] - cost*5
        # 再更新cpu和memory
        self.sat_fea[sat, [SAT_CPU, SAT_MEM]] = self.sat_fea[sat, [SAT_CPU, SAT_MEM]] - api_fea[[API_CPU, API_MEM]]

        # 更新到下一次选择对象
        self.count = self.count + 1
        if self.count < self.input.api_count[0]:
            api = self.api_comb[self.count]

        input_dim = self.sat_all_count * 7 + self.api_fea.shape[1] * 2

        if self.sat_fea[sat, SAT_TMP] > 80 or self.sat_fea[sat, SAT_DELAY] > api_fea[API_DELAY] or torch.min(self.sat_fea[sat,[SAT_CPU, SAT_MEM, SAT_ENG]])<0:
            return self.get_state_concat(api), reward+PUNISH, 1
        if self.count >= self.input.api_count[0]:
            return torch.zeros(input_dim).cuda(), reward, 1
        return self.get_state_concat(api), reward, 0



#用于对sample得到的进行存储
class Memory(object):
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.next_states[:]


# replay buffer
class ReplayBuffer:
    def __init__(self, max_size, batch_size, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0

        self.states = [0] * self.mem_size
        self.actions = [0] * self.mem_size
        self.rewards = [0] * self.mem_size
        self.next_states = [0] * self.mem_size
        self.is_terminals = [0] * self.mem_size
    def store_transition(self, state, action, reward, next_state, done):
        mem_idx = self.mem_cnt % self.mem_size

        self.states[mem_idx] = state
        self.actions[mem_idx] = action
        self.rewards[mem_idx] = reward
        self.next_states[mem_idx] = next_state
        self.is_terminals[mem_idx] = done

        self.mem_cnt += 1
    def get_elements_by_indices(self, original_list, indices):
        return [original_list[i] for i in indices]


    def sample_buffer(self):
        # import ipdb
        # ipdb.set_trace()
        mem_len = min(self.mem_size, self.mem_cnt)

        batch = np.random.choice(mem_len, self.batch_size, replace=False)

        states = self.get_elements_by_indices(self.states, batch)
        actions = self.get_elements_by_indices(self.actions, batch)
        rewards = self.get_elements_by_indices(self.rewards, batch)
        next_states = self.get_elements_by_indices(self.next_states, batch)
        terminals = self.get_elements_by_indices(self.is_terminals, batch)

        return states, actions, rewards, next_states, terminals

    def ready(self):
        return self.mem_cnt > self.batch_size



