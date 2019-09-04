import gym
import torch
import torch.nn as nn
from itertools import count
from torch.distributions import Normal
import numpy as np


class Actor(nn.Module):
    def __init__(self, is_train=True):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.noisy = Normal(0, 0.2)
        self.is_train = is_train

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

    def select_action(self, epsilon, state):
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        with torch.no_grad():
            action = self.forward(state).squeeze() + self.is_train * epsilon * self.noisy.sample()
        return 2 * np.clip(action.item(), -1, 1)


env = gym.make('Pendulum-v0')
actor = Actor(is_train=False).cuda()
actor.load_state_dict(torch.load('ddpg-actor.para'))
epsilon = 1


for epoch in count():
    state = env.reset()
    episode_reward = 0
    env.render()
    for time_step in range(200):
        env.render()
        action = actor.select_action(epsilon, state)
        next_state, reward, done, _ = env.step([action])
        episode_reward += reward
        state = next_state
    print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))


