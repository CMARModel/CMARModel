
import os
from random import choice

import numpy as np
import torch
import glob
import torch.nn as nn
import tqdm
from sklearn.model_selection import ParameterGrid
from torch.utils.tensorboard import SummaryWriter
from utils import Logger

from .evaluator import Evaluator
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from .env import  Env, Memory, ReplayBuffer
torch.autograd.set_detect_anomaly(True)
from collections import namedtuple
import torch.nn.functional as F
from models import *
from utils import *
from torch.distributions import Categorical
import gym

get_train_data_collect = namedtuple('train_data', ['states', 'action_indexs', 'old_probs',
               'adv', 'targets', 'overs'])

def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric + ': ' + str(value)) + ' '
    return result_str


class Learner(object):

    def __init__(self, config, model_cls, loader):
        self.loader = loader
        self.config = config

        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.device)


        env_name = 'LunarLander-v2'
        self.env = gym.make(env_name)
        self.env = self.env.unwrapped
        state_dim = self.env.observation_space.shape[0]
        self.policy = model_cls(config, state_dim)
        # self.env = Env(self.policy)
        # for name, parameter in self.model.named_parameters():
        #     print(name, parameter.requires_grad)

        self.epochs = config.epochs
        self.eval_per_epoch = [6, 4, 4, 4, 2, 2]
        self.early_stop_rounds = config.early_stop
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay

        self.cur_epoch = 0
        self.best_epoch = 0
        self.best_result = np.inf
        self.valid_per_epoch = 5
        self.rounds = self.config.batch_train_round
        self.cur_round = 0
        self.log_interval = 50

        # if hasattr(nn, config.loss):
        #     self.loss_func = getattr(nn, config.loss)(reduction=config.reduction)
        # elif config.loss == 'BPRLoss':
        #     self.loss_func = BPRLoss()
        self.save_floder = os.path.join(config.savepath, config.dataset, config.model)
        self.version = self.gen_version()
        self.logpath = os.path.join(self.save_floder, f"version_{self.version}")
        # import ipdb
        # ipdb.set_trace()
        self.writer = SummaryWriter(self.logpath)
        self.train_writer_count = 0
        self.eval_writer_count = 0
        self.logger = Logger(os.path.join(self.logpath, 'run.log'))
        self.logger.info(f'lr: {self.learning_rate }')

        self.evaluator = Evaluator(metrics=config.metrics, topk=config.topk,
                                   case_study=config.case_study,
                                   savepath=self.logpath)

        self.main_metric = config.main_metric
        self.param_space = config.grid_search



    def action_prob_mask(self, unmasklogit):
        # 对不符合要求的action对应的prob进行mask
        api = self.env.api_comb[self.env.count]
        # 先计算得到假设选取任一动作的卫星状态[action_space, sat_all_count, 5]
        sat_fea_space = self.env.sat_fea.clone()
        sat_fea_space = sat_fea_space.reshape(1, self.policy.sat_count + 1, 5).repeat(self.env.action_space, 1, 1)

        mask = torch.eye(self.env.action_space, self.policy.sat_count + 1).cuda()
        cost = self.policy.cost_info[api].reshape(1, -1).repeat(self.env.action_space, 1)
        cost = torch.mul(cost, mask)

        api_fea = self.policy.api_info[api].reshape(1, 1, -1).repeat(self.env.action_space, self.policy.sat_count+1, 1)
        fea = torch.mul(api_fea, mask.reshape(self.env.action_space, self.policy.sat_count + 1, 1).repeat(1, 1, 3))

        sat_fea_space[:, :, SAT_ENG] = sat_fea_space[:, :, SAT_ENG] - cost
        sat_fea_space[:, :, [SAT_CPU, SAT_MEM]] = sat_fea_space[:, :, [SAT_CPU, SAT_MEM]] - fea[:, :, [API_CPU, API_MEM]]

        sat_fea_space[:, :, SAT_DELAY] = fea[:, :, API_DELAY] - sat_fea_space[:, :, SAT_DELAY]
        sat_fea_space[:, :, SAT_DELAY] = sat_fea_space[:, :, SAT_DELAY] * mask

        sat_fea_space[:, :, SAT_TMP] = 80 * torch.eye(self.env.action_space, self.policy.sat_count + 1).cuda() - sat_fea_space[:, :, SAT_TMP]*mask

        sat_fea_space_min = torch.min(sat_fea_space.reshape(self.env.action_space,  -1), dim=-1)[0]

        masklogit = torch.where(sat_fea_space_min>=0, unmasklogit,
                               torch.full([self.env.action_space], -np.inf).cuda().float())

        masklogit = torch.where(self.policy.cost_info[api] >= 0, masklogit,
                               torch.full([self.env.action_space], -np.inf).cuda().float())

        return masklogit

    def sample_action(self, state):
        unmasklogit = self.policy.q_eval(state)
        logit = unmasklogit
        # logit = self.action_prob_mask(unmasklogit)
        action = torch.argmax(logit).item()

        if np.random.random() < self.policy.epsilon:
            action = np.random.choice(np.arange(1, self.policy.sat_count+1))

        return action

    def get_best_action(self, state):
        unmasklogit = self.policy.q_eval(state)
        logit = self.action_prob_mask(unmasklogit)
        action = torch.argmax(logit).item()

        return action

    def get_best_action_unmask(self, state):
        unmasklogit = self.policy.q_eval(state)
        logit = self.action_prob_mask(unmasklogit)
        action = torch.argmax(logit).item()

        return action

    def sample_action_prob_mask(self, unmasklogit):

        masklogit = self.action_prob_mask(unmasklogit)

        dist = Categorical(logits=masklogit)
        action = dist.sample()

        # self.memory.actions.append(action)
        # self.memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def sample_action_prob_unmask(self, unmasklogit):

        dist = Categorical(logits=unmasklogit)
        action = dist.sample()
        #
        # self.memory.actions.append(action)
        # self.memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def get_best_action_prob_mask(self, unmasklogit):
        masklogit = self.action_prob_mask(unmasklogit)

        action = torch.argmax(masklogit[1:])+1

        return action.item()

    def get_best_action_prob_unmask(self, unmasklogit):
        unmasklogit = unmasklogit
        action = torch.argmax(unmasklogit[1:])+1

        return action.item()


    def get_model_param_log(self):
        for name, param in self.model.actor_net.named_parameters():
            if param.grad is not None:
                self.logger.info(f'epoch: {self.cur_epoch} name: {name} param:{torch.mean(param)}  grad:{torch.mean(param.grad)}')
        for name, param in self.model.critic_net.named_parameters():
            if param.grad is not None:
                self.logger.info(f'epoch: {self.cur_epoch} name: {name} param:{torch.mean(param)}  grad:{torch.mean(param.grad)}')

    def train_one_epoch(self):
        total_train_loss = 0.0
        total_match_count = 0
        total_infer_avail_count = 0
        total_infer_cost_mean = 0.0
        i_sample = 0
        running_reward = 0
        avg_length = 0
        timestep = 0
        with tqdm.tqdm(self.loader.train_dataloader(), ncols=100) as pbar:
            for step_raw, input_raw in enumerate(pbar):
                i_sample += 1
                input_raw = input_raw.cuda() #[api_comb_max_length]
                #类似于env.reset()
                state = self.env.reset()[0]
                state = torch.FloatTensor(state).cuda()

                # sample data, 一次sample_data对一条数据进行sample
                for t in range(self.config.api_comb_max_length):
                    timestep += 1

                    # import ipdb
                    # ipdb.set_trace()
                    action = self.sample_action(state)

                    # env中记录了当前state原始情况，返回的state是embedding向量
                    # 返回的action和其他框架对齐，只返回id, 其他action背景信息放在env中
                    # 返回的action和其他框架对齐，只返回id, 其他action背景信息放在env中
                    next_state, reward, over, _ = self.env.step(action)

                    self.policy.memory.store_transition(state, action, reward, next_state, over)
                    if self.policy.memory.ready():
                        self.policy.update()

                        # self.get_model_param_log()

                    running_reward += reward

                    if over:
                        break

                    state = next_state

                avg_length += t

                if i_sample % self.log_interval == 0:
                    avg_length = avg_length / self.log_interval
                    running_reward = (running_reward / self.log_interval)
                    self.logger.info(
                        'training Episode {} \t avg length: {} \t reward: {}'.format(i_sample, avg_length, running_reward))
                    self.writer.add_scalar(
                        f'train/reward', running_reward, global_step=self.train_writer_count)
                    self.train_writer_count += 1
                    running_reward = 0
                    avg_length = 0

                if ((self.cur_epoch < len(self.eval_per_epoch)) and (step_raw % (len(pbar) // self.eval_per_epoch[self.cur_epoch]) == 1) and step_raw > 2):
                    self.valid()
                # if step % 10 ==1:
                    # self.logger.info(f'after backword')

            # if self.cur_epoch % 30 == 1:
            


    def train(self):
        while self.cur_epoch < self.epochs:
            self.train_one_epoch()
            metrics = self.valid()
            # if metrics[self.main_metric] < self.best_result:
            #     self.best_result = metrics[self.main_metric]
            #     self.best_epoch = self.cur_epoch
            #     self.save()

            # if self.cur_epoch - self.best_epoch > self.early_stop_rounds:
            #     self.logger.info("early stop...")
            #     break
            # self.logger.info(f'vaild result:  epoch : {self.cur_epoch}, test metrics: {dict2str(metrics)}')
            self.cur_epoch += 1


    def valid(self):
        metrics = self.eval_one_epoch(loader=self.loader.valid_dataloader())
        self.logger.info(f'epoch: {self.cur_epoch} valid metrics: {dict2str(metrics)}')
        self.writer.add_scalar(
            f'eval/avg_rewards', metrics['total_reward'], global_step=self.eval_writer_count)
        self.writer.add_scalar(
            f'eval/avg_avail_rate', metrics['unavail rate'], global_step=self.eval_writer_count)
        self.eval_writer_count += 1
        return metrics

    @torch.no_grad()
    def case_study(self):
        setattr(self.model, 'result', [])
        self.model.eval()
        choices = []
        pos_idxs, pos_lens = [], []
        for input in self.loader.test_dataloader():
            input = input.cuda()
            scores = self.model.predict(input)
            c = self.model.idx.gather(dim=-1, index=input.target_ids.view(-1,1)).cpu().numpy()
            choices.append(c)
            pos_idx, pos_len = self.evaluator.collect(scores, input)
            pos_idxs.append(pos_idx)
            pos_lens.append(pos_len)
        pos_idx = np.concatenate(pos_idxs, axis=0).astype(bool)
        pos_len = np.concatenate(pos_lens, axis=0)
        # get metrics
        result_matrix = self.evaluator._calculate_metrics(pos_idx, pos_len)
        result = np.stack(result_matrix, axis=0)[0, :, 0]
        result = result.nonzero()[0]
        choices = np.concatenate(choices).flatten()
        # choices = choices[result]
        np.save(f'{self.config.dataset}.npy', choices)

    def test(self):
        self.load()
        metrics = self.eval_one_epoch(loader=self.loader.test_dataloader())
        # for key, value in metrics.items():
        #     self.writer.add_scalar(
        #         f'test/{key}', value, global_step=self.cur_epoch)

        self.logger.info(f'best epoch : {self.best_epoch}, test metrics: {dict2str(metrics)}')
        return metrics

    @torch.no_grad()
    def eval_one_epoch(self, loader):
        self.policy.q_eval.eval()
        ep_reward = 0.0
        i_sample = 0
        result = {}
        avail_count = 0
        unavail_count = 0
        best_reward = []
        for step, input_raw in enumerate(loader):
            input_raw = input_raw.cuda()
            i_sample += 1
            sample_reward = 0
            state = self.env.reset(input_raw)
            sat_choose_list = []
            sat_choose_list_unmask = []
            for t in range(self.config.api_comb_max_length):
                # state = state.repeat(self.config.train_batch_size, 1)
                # state = state.reshape(1, -1)
                action = self.get_best_action(state)
                action_unmask = self.get_best_action_unmask(state)
                sat_choose_list.append(action)
                sat_choose_list_unmask.append(action_unmask)
                next_state, reward, over = self.env.step(action)

                sample_reward += reward
                # if render:
                #     env.render()
                if over:
                    if t == (input_raw.api_count[0] - 1):
                        avail_count += 1
                        ep_reward += sample_reward
                        best_reward.append(input_raw.label.item())
                    else:
                        unavail_count += 1
                    break
                state = next_state
            if step <=50:
                self.logger.info(f'epoch : {self.cur_epoch}, step:{step}, test sat choose:{sat_choose_list},test sat choose unmask:{sat_choose_list_unmask}, truth sat choose:{input_raw.opt_sat_comb}, test_reward:{sample_reward}, truth_reward:{input_raw.label.item()}')
                # print(f'epoch : {self.cur_epoch}, step:{step}, test sat choose:{sat_choose_list},truth sat choose:{input_raw.opt_sat_comb}, test_reward:{sample_reward}, truth_reward:{input_raw.label.item()}')
        # print('epoch: {}\tReward: {}\tunavail rate:{}'.format(self.cur_epoch, ep_reward/avail_count, unavail_count/(unavail_count+avail_count)))

        # self.eval_sampler = Sampler(self.model)
        # cost_sum = 0
        # avail_list = []
        # for step, input in enumerate(loader):
        #     input_data = input.cuda()
        #     cost, avail = self.eval_sampler.get_best_reward(self.logger, input_data)
        #     cost_sum += cost
        #     avail_list.append(avail)
        #
        # result['cost_mean'] = cost_sum / len(loader)
        # result['avail_mean'] = torch.mean(torch.cat(avail_list,dim =-1))
        # # result["arg_sat"] = comb_result
        result['total_reward'] = ep_reward / (avail_count+1e-10)
        result['unavail rate'] = unavail_count/(unavail_count+avail_count)
        result['best_reward'] = np.mean(best_reward)
        return result

    def save(self):
        filename = glob.glob(os.path.join(self.save_floder, f"version_{self.version}", '*.pth'))
        if filename:
            os.remove(filename[0])
        filename = os.path.join(self.save_floder, f"version_{self.version}", f"epoch={self.cur_epoch}.pth")
        state = {
            'best_epoch': self.best_epoch,
            'cur_epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
        }
        torch.save(state, filename)

    def load(self):
        filename = glob.glob(os.path.join(self.save_floder, f"version_{self.version}", '*.pth'))[0]
        state = torch.load(filename)
        self.cur_epoch = state['cur_epoch']
        self.best_epoch = state['best_epoch']
        self.model.load_state_dict(state['state_dict'])

    def gen_version(self):
        dirs = glob.glob(os.path.join(self.save_floder, '*'))
        if not dirs:
            return 0
        if self.config.version < 0:
            version = max([int(x.split(os.sep)[-1].split('_')[-1])
                           for x in dirs]) + 1
        else:
            version = self.config.version
        return version