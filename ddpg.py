import gym
import torch
import torch.nn as nn
from itertools import count
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256 + 1, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x, a):
        x = self.relu(self.fc1(x))
        x = torch.cat((x, a), dim=1)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.forward(state).squeeze() + self.is_train * epsilon * self.noisy.sample()
        return 2 * np.clip(action.item(), -1, 1)


class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


env = gym.make('Pendulum-v0')
actor = Actor().to(device)
critic = Critic().to(device)
actor_target = Actor().to(device)
critic_target = Critic().to(device)

critic_optim = torch.optim.Adam(critic.parameters(), lr=3e-5)
actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-5)

explore = 50000
epsilon = 1
gamma = 0.99

memory_replay = Memory(50000)
begin_train = False
batch_size = 32

learn_steps = 0

writer = SummaryWriter('logs')

for epoch in count():
    state = env.reset()
    episode_reward = 0
    for time_step in range(200):
        action = actor.select_action(epsilon, state)
        next_state, reward, done, _ = env.step([action])
        episode_reward += reward
        reward = (reward + 8.1) / 8.1
        memory_replay.add((state, next_state, action, reward))
        if memory_replay.size() > 1280:
            if learn_steps % 4 == 0:
                actor_target.load_state_dict(actor.state_dict())
                critic_target.load_state_dict(critic.state_dict())
            learn_steps += 1
            if not begin_train:
                print('train begin!')
                begin_train = True
            experiences = memory_replay.sample(batch_size, False)
            batch_state, batch_next_state, batch_action, batch_reward = zip(*experiences)

            batch_state = torch.FloatTensor(batch_state).to(device)
            batch_next_state = torch.FloatTensor(batch_next_state).to(device)
            batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
            batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)

            # print(batch_state.shape, batch_next_state.shape, batch_action.shape, batch_reward.shape)

            with torch.no_grad():
                Q_next = critic_target(batch_next_state, actor_target(batch_next_state))
                Q_target = batch_reward + gamma * Q_next

            critic_loss = F.mse_loss(critic(batch_state, batch_action), Q_target)

            # print(Q_target.shape)

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            writer.add_scalar('critic loss', critic_loss.item(), learn_steps)
            critic.eval()
            actor_loss = - critic(batch_state, actor(batch_state))
            # print(actor_loss.shape)
            actor_loss = actor_loss.mean()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()
            critic.train()
            writer.add_scalar('actor loss', actor_loss.item(), learn_steps)

        if epsilon > 0:
            epsilon -= 1 / explore
        state = next_state

    writer.add_scalar('episode reward', episode_reward, epoch)
    if epoch % 10 == 0:
        print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))

    if epoch % 200 == 0:
        torch.save(actor.state_dict(), 'ddpg-actor' + str(epoch) + '.para')
        torch.save(critic.state_dict(), 'ddpg-critic' + str(epoch) + '.para')
        print('model saved!')



