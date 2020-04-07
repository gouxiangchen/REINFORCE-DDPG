import gym
import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter
from itertools import count
from collections import deque
import time
from tensorflow.keras import Model, layers, optimizers


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


class Actor(Model):
	def __init__(self, trainable=True):
		super(Actor, self).__init__()
		self.fc1 = layers.Dense(256, activation=tf.nn.relu, trainable=trainable)
		self.fc2 = layers.Dense(128, activation=tf.nn.relu, trainable=trainable)
		self.fc3 = layers.Dense(1, activation=tf.nn.tanh, trainable=trainable)

	def call(self, x):
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		return x

	def select_action(self, state, epsilon):
		state = state[np.newaxis, :]
		action = tf.stop_gradient(self.call(state))
		action = np.squeeze(action) + epsilon * np.random.normal(0, 0.2)
		return 2 * np.clip(action, -1, 1)


class Critic(Model):
	def __init__(self, trainable=True):
		super(Critic, self).__init__()
		self.fc1 = layers.Dense(256, activation=tf.nn.relu, trainable=trainable)
		self.fc2 = layers.Dense(128, activation=tf.nn.relu, trainable=trainable)
		self.fc3 = layers.Dense(1, trainable=trainable)

	def call(self, x, a):
		x = self.fc1(x)
		x = tf.concat([x, a], axis=1)
		x = self.fc2(x)
		x = self.fc3(x)
		return x


def soft_update(target, source, tau):
	target_weights = target.get_weights()
	source_weights = source.get_weights()

	for i in range(len(target_weights)):
		target_weights[i] = target_weights[i] * (1.0 - tau) + source_weights[i] * tau
	target.set_weights(target_weights)


explore = 50000
epsilon = 1
gamma = 0.99
tau = 0.005
begin_train = False
batch_size = 32

learn_steps = 0


if __name__ == '__main__':
	tf.keras.backend.clear_session()
	tf.keras.backend.set_floatx('float64')

	memory_replay = Memory(50000)
	writer = SummaryWriter('td3-tf2-logs')
	actor = Actor()
	target_actor = Actor(False)
	target_actor.set_weights(actor.get_weights())

	critic_1 = Critic()
	critic_2 = Critic()

	critic_1_target = Critic(False)
	critic_2_target = Critic(False)
	critic_1_target.set_weights(critic_1.get_weights())
	critic_2_target.set_weights(critic_2.get_weights())

	critic_1_optim = optimizers.Adam(3e-5)
	critic_2_optim = optimizers.Adam(3e-5)
	actor_optim = optimizers.Adam(1e-5)

	mse = tf.keras.losses.MeanSquaredError()

	env = gym.make('Pendulum-v0')
	for epoch in count():
		state = env.reset()
		episode_reward = 0
		for time_step in range(200):
			# start = time.time()
			action = actor.select_action(state, epsilon)
			# end = time.time()
			# print('select action : ', end - start)
			next_state, reward, done, _ = env.step([action])
			episode_reward += reward
			reward = (reward + 8.1) / 8.1
			memory_replay.add((state, next_state, action, reward))
			if memory_replay.size() > 1280:
				learn_steps += 1
				if not begin_train:
					print('train begin!')
					begin_train = True
				experiences = memory_replay.sample(batch_size, False)
				batch_state, batch_next_state, batch_action, batch_reward = zip(*experiences)

				batch_state = np.asarray(batch_state)
				batch_next_state = np.asarray(batch_next_state)
				batch_action = np.asarray(batch_action)
				batch_reward = np.asarray(batch_reward)
				# print('action : ', batch_action, 'reward : ', batch_reward)
				batch_reward = batch_reward[:, np.newaxis]
				batch_action = batch_action[:, np.newaxis]

				# print(batch_state.shape, batch_next_state.shape, batch_action.shape, batch_reward.shape)

				a_ = target_actor(batch_next_state)
				q_next_1 = critic_1_target(batch_next_state, a_)
				q_next_2 = critic_2_target(batch_next_state, a_)
				q = tf.minimum(q_next_1, q_next_2)
				q_target = tf.stop_gradient(batch_reward + gamma * q)

				with tf.GradientTape() as g:
					q1 = critic_1(batch_state, batch_action)
					critic_1_loss = tf.reduce_mean((q1 - q_target) ** 2)

				critic_1_grads = g.gradient(critic_1_loss, critic_1.trainable_variables)
				critic_1_optim.apply_gradients(zip(critic_1_grads, critic_1.trainable_variables))

				with tf.GradientTape() as g:
					q2 = critic_2(batch_state, batch_action)
					critic_2_loss = mse(q2, q_target)

				critic_2_grads = g.gradient(critic_2_loss, critic_2.trainable_variables)
				critic_2_optim.apply_gradients(zip(critic_2_grads, critic_2.trainable_variables))

				writer.add_scalar('critic loss', float(critic_1_loss), learn_steps)
				writer.add_scalar('critic 2 loss', float(critic_2_loss), learn_steps)

				if learn_steps % 2 == 0:
					# a_weight_0 = np.asarray(actor.get_weights()[0])
					# with tf.GradientTape(watch_accessed_variables=False) as g3:
					# 	g3.watch(actor.trainable_variables)
					# 	a = actor(batch_state)
					# 	actor_loss = -1.0 * tf.reduce_mean(critic_1(batch_state, a))

					with tf.GradientTape() as g:
						a = actor(batch_state)
						actor_loss = -1.0 * tf.reduce_mean(critic_1(batch_state, a))


					actor_grads = g.gradient(actor_loss, actor.trainable_variables)

					actor_optim.apply_gradients(zip(actor_grads, actor.trainable_variables))

					# a_weight_0_hat = np.asarray(actor.get_weights()[0])

					# print('actor improved : ', np.linalg.norm(a_weight_0_hat - a_weight_0))
					writer.add_scalar('actor loss', float(actor_loss), learn_steps/2)

					soft_update(target_actor, actor, tau)
					soft_update(critic_1_target, critic_1, tau)
					soft_update(critic_2_target, critic_2, tau)

					
					

			if epsilon > 0:
				epsilon -= 1 / explore
			state = next_state
		if epoch % 1 == 0:
			print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))
		if (epoch + 1) % 10 == 0:
			actor.save_weights('td3_model/tf2_td3.h5')
			print('model saved!')








