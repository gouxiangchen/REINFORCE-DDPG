import gym
import torch
import torch.nn as nn
from itertools import count
from torch.distributions import Bernoulli


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
policy.load_state_dict(torch.load('cart-policy.para'))

for epoch in count():
    episode_reward = 0
    state = env.reset()
    for time_step in range(200):
        env.render()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = int(policy.select_action(state))
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state
        if done:
            break

    print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))






