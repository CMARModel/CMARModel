import os
import pickle
from functools import partial
from turtle import left

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
from torch.nn.utils.rnn import pad_sequence


class Input(object):

    def __init__(self, **kwgs):
        for key, value in kwgs.items():
            value = torch.stack(value, dim=0)
            setattr(self, key, value)

    def cuda(self):
        for key in self.__dict__:
            setattr(self, key, getattr(self, key).cuda())
        return self


class TheDataLoader(object):

    def __init__(self, dataset='random', root='.', train_sample_size=128, eval_batch_size=1024,
                 num_workers=4, negative_sample=-1, candidate_sample=False, api_comb_max_length=15):
        self.dataset = dataset
        self.root = root
        self.loadpath = os.path.join(self.root, self.dataset)
        self.train_sample_size = train_sample_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.api_comb_max_length = api_comb_max_length
        self._build_dataset()

    def _build_dataset(self):
        # with open(os.path.join(self.loadpath, f'service_api.pkl'), 'rb') as f:
        #     self.info = pickle.load(f)
        # self.n_items = self.info['item_num']
        self.train_data = TheDataset(dataset=self.dataset, root=self.root, file = '30_train_input.csv',
                                            phase='train')
        self.valid_data = TheDataset(dataset=self.dataset, root=self.root, file = '30_valid_input.csv',
                                            phase='valid')
        # self.valid_data = TheDataset(dataset=self.dataset, root=self.root, file='30_train_input.csv',
        #                              phase='valid')

        self.test_data = TheDataset(dataset=self.dataset, root=self.root, file='30_test_input.csv',
                                    phase='test')

        # self.test_data = TheDataset(dataset=self.dataset, root=self.root, file='30_train_input.csv',
        #                             phase='test')

    def pad_sequence2fixlen(self, data, length):
        data = data+[torch.tensor([0 for i in range(length)])]
        data_pad = pad_sequence(data, batch_first=True, padding_value=0)
        return data_pad[:-1, :]

    def _collate_fn(self, data):
        combination = [tup[0] for tup in data]
        comb_pad = self.pad_sequence2fixlen(combination, self.api_comb_max_length)
        opt_comb = [tup[4] for tup in data]
        opt_comb_pad = self.pad_sequence2fixlen(opt_comb, self.api_comb_max_length)
        opt_comb_loc = [tup[5] for tup in data]
        opt_comb_loc_pad = self.pad_sequence2fixlen(opt_comb_loc, self.api_comb_max_length)
        # 'api_count': torch.tensor(len(pad[0])),
        feed_dict = {
            'api_comb': [torch.as_tensor(i) for i in comb_pad],
            'api_count': [tup[1] for tup in data],
            'sat_fea': [tup[2] for tup in data],
            'label': [tup[3] for tup in data],
            'opt_sat_comb': [torch.as_tensor(i) for i in opt_comb_pad],
            'opt_sat_comb_loc': [torch.as_tensor(i) for i in opt_comb_loc_pad]
        }        
        return Input(**feed_dict)


    def train_dataloader(self):
        # 此处是sample的batch,每次只对一条数据进行sample
        return DataLoader(dataset=self.train_data,
                          batch_size=self.train_sample_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          collate_fn=partial(self._collate_fn))

    def valid_dataloader(self):
        return DataLoader(dataset=self.valid_data,
                          batch_size=self.eval_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=partial(self._collate_fn))

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data,
                          batch_size=self.eval_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=partial(self._collate_fn))


class TheDataset(Dataset):

    def __init__(self, dataset='random', root='.', file='30_valid_input.csv', phase='train') -> None:
        self.loadpath = os.path.join(root, dataset)
        self.dataset = dataset
        self.phase = phase
        self.file = file

        self._load_data()


    def _load_data(self):
        # self.user_ids = []
        # self.api_combination = []
        # self.api_count = []
        # self.sat_fea_info = []
        # self.label = []
        # sat_sta = pd.read_pickle(os.path.join(self.loadpath, f'satellite_info.pickle'))
        data = pd.read_csv(os.path.join(self.loadpath, self.file))
        self.api_combination = [torch.tensor(eval(i)) for i in data['api_comb'].tolist()]
        self.api_count = [torch.tensor(len(eval(i))) for i in data['api_comb'].tolist()]
        self.sat_fea_info = [torch.tensor(eval(i)) for i in data['sat_fea'].tolist()]
        self.label = [torch.tensor(i) for i in data['label'].tolist()]
        self.opt_sat_comb =  [torch.tensor(eval(i)) for i in data['opt_sat_comb'].tolist()]
        self.opt_sat_comb_loc = [torch.tensor(eval(i)) for i in data['opt_sat_comb_loc'].tolist()]

    def __len__(self):
        return len(self.api_combination)

    def __getitem__(self, idx):
        # import pdb
        # pdb.set_trace()
        if self.phase in ['valid', 'test']:
            return self.api_combination[idx], self.api_count[idx], self.sat_fea_info[idx], self.label[idx], self.opt_sat_comb[idx], self.opt_sat_comb_loc[idx]
        else:
            return self.api_combination[idx], self.api_count[idx], self.sat_fea_info[idx], self.label[idx], self.opt_sat_comb[idx], self.opt_sat_comb_loc[idx]

# if __name__ == "__main__":
#     loader = TheDataLoader('random', '../dataset', negative_sample=10)

#     for seq_input in tqdm(loader.train_dataloader()):
#         seq_input = seq_input.cuda()
#         print(seq_input.__dict__)
#         breakcd

#     for seq_input in tqdm(loader.valid_dataloader()):
#         seq_input = seq_input.cuda()
#         print(seq_input.__dict__)
#         break

#     for seq_input in tqdm(loader.test_dataloader()):
#         seq_input = seq_input.cuda()
#         print(seq_input.__dict__)
#         break
