import random
import sys
import time
import logging
import argparse
import itertools
from six import StringIO
from random import sample, randint
import pylab as pl

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
import Game2048_Env


class Net(nn.Module):
    def __init__(self, obs, n_actions):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(obs, 128, kernel_size=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=2)
        self.fc1 = nn.Linear(16, n_actions)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.fc1(x.view(x.shape[0], -1))  # todo
        return x


LEARNING_RATE = 0.0001
BATCH_SIZE = 256
TARGET_UPDATE_FRQ = 50
GAMMA = 0.99
EPSILON = 0.9
MEMORY_SIZE = 2000


# EPOCHS = 2000


class Memory:
    def __init__(self, memory_size, n_obs):
        self.s = np.zeros(n_obs, dtype=np.float32)
        self.s_ = np.zeros(n_obs, dtype=np.float32)
        self.a = np.zeros(memory_size, dtype=np.int32)
        self.r = np.zeros(memory_size, dtype=np.float32)
        self.done = np.zeros(memory_size, dtype=np.float32)

        # replaybuffer大小
        self.buffer_size = memory_size
        self.size = 0
        self.pos = 0

    def store_memory(self, s, a, s_, done, r):
        self.s[self.pos] = s
        self.a[self.pos] = a
        if not done:
            self.s_[self.pos] = s_
        self.done[self.pos] = done
        self.r[self.pos] = r

        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def get_sample(self, size):
        index = sample(range(0, self.size), size)
        return self.s[index], self.a[index], self.s_[index], self.done[index], self.r[index]


class DQN:
    def __init__(self, n_obs, n_action, EPSILON):
        self.eval_model, self.target_model = Net(n_obs, n_action), Net(n_obs, n_action)
        self.optimizer = torch.optim.Adam(self.eval_model.parameters(), LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.n_action = n_action
        self.step_counter = 0
        self.options = list(range(n_action))
        self.epsilon = EPSILON
        self.record_loss = []

    def decay_learning_rate(self):
        for params in self.optimizer.param_groups:  # 遍历Optimizer中的每一组参数
            params['lr'] *= 0.6  # 将该组参数的学习率 * 0.9
        print("decay learning rate")

    def learn(self, memory):
        if memory.size >= BATCH_SIZE:
            if self.step_counter % TARGET_UPDATE_FRQ == 0:
                self.target_model.load_state_dict(self.eval_model.state_dict())
            self.step_counter += 1

            s, a, s_, done, r = memory.get_sample(BATCH_SIZE)
            s = torch.FloatTensor(s)
            s_ = torch.FloatTensor(s_)
            r = torch.FloatTensor(r)
            a = torch.LongTensor(a)

            target_q = r + torch.FloatTensor(GAMMA * (1 - done)) * self.target_model(s_).max(1)[0]
            target_q = target_q.view(BATCH_SIZE, 1)
            # print("a: ",torch.reshape(a, shape=(a.size()[0], -1)))
            eval_q = self.eval_model(s).gather(1, torch.reshape(a, shape=(a.size()[0], -1)))
            loss = self.loss(eval_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.record_loss.append(np.array(loss.item()).mean())
            self.optimizer.step()

    def get_action(self, s, changed=True):
        # if np.random.uniform() >= EPSILON:
        #     action = randint(0, self.n_action - 1)
        # else:
        #     q = self.eval_model(torch.FloatTensor(s))
        #     m, index = torch.max(q, 1)
        #     action = index.data.numpy()[0]
        if changed:
            self.options = list(range(self.n_action))  # 复原
            # print("restore")
        q = self.eval_model(torch.FloatTensor(s))
        m, index = torch.max(q, 1)
        model_action = index.data.numpy()[0]
        action = random.choice(self.options) if np.random.uniform() <= self.epsilon or not changed else model_action
        self.options.remove(action)
        return action

    def train_ai(self):
        begin_t = time.time()
        max_reward = 0
        max_num = 0
        avg_num = 0
        i_episode = 0
        # loss = []
        while 1:
            i_episode += 1
            # 每局开始，重置环境
            s = env.reset()
            # 累计奖励值
            ep_r = 0
            changed = True
            while True:

                # 计算动作 (1,4,4,1)
                a = dqn.get_action(np.expand_dims(s, axis=0), changed)
                # 执行动作
                s_, r, done, info, changed = env.step(a)
                # 存储信息
                memory.store_memory(s, a, s_, done, r)
                # 学习优化过程
                dqn.learn(memory)
                if done:
                    avg_num += info["highest"]

                    # env.render()
                    if env.score > max_reward:
                        max_reward = env.score
                        print("current_max_reward ", max_reward)
                        # # 保存模型
                        # torch.save(dqn.eval_model, "2048.pt")
                        if info["highest"] > max_num:
                            max_num = info["highest"]
                            if 512 >= max_num >= 128: dqn.decay_learning_rate()
                            if max_num > 256: self.epsilon = 0.95
                            print("max_num:  ", max_num)
                    # print('Ep: ', i_episode,
                    #       '| num: ', info["highest"],
                    #       '| current top scores: ', max_reward,
                    #       '| scores: ', env.score)
                    break
                s = s_
                if info["highest"] == 2048:
                    print("finish! time cost is {}s".format(time.time() - begin_t))
                    return
            if i_episode % 100 == 0:
                print('Ep: ', i_episode,
                      '| highest: ', max_num,
                      '| current top scores: ', max_reward,
                      '| avg num: ', avg_num / 100)
                avg_num = 0


env = Game2048_Env.Game2048Env()

dqn = DQN(16, env.n_actions, EPSILON)
memory = Memory(MEMORY_SIZE, (MEMORY_SIZE, 4, 4, 16))
dqn.train_ai()
