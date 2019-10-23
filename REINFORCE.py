import gym
import torch
import torch.nn as nn
from itertools import count
from torch.distributions import Bernoulli
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

    def select_action(self, state):
        with torch.no_grad():
            prob = self.forward(state)
            b = Bernoulli(prob)
            action = b.sample()
        return action.item()


env = gym.make('CartPole-v0')
policy = PolicyNetwork().to(device)
gamma = 0.99
optim = torch.optim.Adam(policy.parameters(), lr=1e-4)


for epoch in count():
    episode_reward = 0
    state = env.reset()
    rewards = []
    actions = []
    states = []
    for time_step in range(200):
        states.append(state)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = int(policy.select_action(state))
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state
        rewards.append(reward)
        if done:
            break

    R = 0
    for i in reversed(range(len(rewards))):
        R = gamma * R + rewards[i]
        rewards[i] = R

    rewards_mean = np.mean(rewards)
    rewards_std = np.std(rewards)
    rewards = (rewards - rewards_mean) / rewards_std

    states_tensor = torch.FloatTensor(states).to(device)
    actions_tensor = torch.FloatTensor(actions).unsqueeze(1).to(device)
    rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)

    # print(states_tensor.shape, actions_tensor.shape, rewards_tensor.shape)

    prob = policy(states_tensor)
    # print(prob.shape)
    b = Bernoulli(prob)
    log_prob = b.log_prob(actions_tensor)
    # print(log_prob.shape)
    loss = -log_prob * rewards_tensor
    # print(loss.shape)
    loss = loss.mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    if epoch % 10 == 0:
        print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))
        torch.save(policy.state_dict(), 'cart-policy.para')






