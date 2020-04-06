from TD3_tf1 import DDPG
import gym
import tensorflow as tf
import numpy as np
from itertools import count
from collections import deque
import time


if __name__ == '__main__':
	ddpg = DDPG()

	with tf.Session() as sess:

		ddpg.load_model(sess)
		env = gym.make('Pendulum-v0')
		for epoch in count():
			state = env.reset()
			episode_reward = 0
			for time_step in range(200):
				env.render()
				# start = time.time()
				action = ddpg.select_action(sess, state, 0)
				# end = time.time()
				# print('select action : ', end - start)
				next_state, reward, done, _ = env.step([action])
				episode_reward += reward
				reward = (reward + 8.1) / 8.1
				state = next_state

			print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))


