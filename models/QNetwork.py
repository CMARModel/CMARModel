import os
import pickle
import math
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
torch.set_printoptions(threshold=np.inf)
torch.autograd.set_detect_anomaly(True)

class QNetwork(nn.Module):
    def __init__(self, state_dim, weight_decay, learning_rate, loadpath='./dataset/v2_fix/', dataset='v2_fix', embedding_size=128, hidden_size=128, api_comb_max_length=15,
                 sat_all_count=31, api_all_count=201, sat_fea_count=5, api_fea_count=3):
        super(QNetwork, self).__init__()
        self.embedding_size = embedding_size
        self.sat_all_count = sat_all_count
        self.api_fea_count = api_fea_count
        input_dim = self.sat_all_count * 7 + self.api_fea_count * 2
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        # input_dim = self.sat_all_count
        input_dim = state_dim
        self.actor_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            # torch.nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, self.embedding_size)
        )

        self.actor_head = nn.Sequential(
            nn.Linear(self.embedding_size, 4),
            # torch.nn.BatchNorm1d(128),
            # nn.Linear(128, self.sat_all_count)
        )
        self.optimizer = torch.optim.Adam(self.parameters(),weight_decay=self.weight_decay, lr=self.learning_rate)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        if isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            # TODO 初始化check
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input):
        actor_embed = self.actor_encoder(input)
        actor_output = self.actor_head(actor_embed)
        return actor_output
