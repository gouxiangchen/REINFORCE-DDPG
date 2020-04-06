import gym
import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter
from itertools import count
from collections import deque
import time


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

explore = 50000
epsilon = 1
gamma = 0.99
tau = 0.005
begin_train = False
batch_size = 32

learn_steps = 0


class DDPG:
	def __init__(self):
		self.batch_state = tf.placeholder(tf.float32, [None, 3], name='state')
		self.batch_next_state = tf.placeholder(tf.float32, [None, 3], name='next_state')
		with tf.variable_scope('online_actor'):
			actor_fc1 = tf.layers.dense(self.batch_state, 256, tf.nn.relu, name='fc1')
			actor_fc2 = tf.layers.dense(actor_fc1, 128, tf.nn.relu, name='fc2')
			self.actor_out = tf.layers.dense(actor_fc2, 1, tf.nn.tanh, name='out')

		with tf.variable_scope('target_actor'):
			actor_target_fc1 = tf.layers.dense(self.batch_next_state, 256, tf.nn.relu, name='fc1', trainable=False)
			actor_target_fc2 = tf.layers.dense(actor_target_fc1, 128, tf.nn.relu, name='fc2', trainable=False)
			self.actor_target_out = tf.layers.dense(actor_target_fc2, 1, tf.nn.tanh, name='out', trainable=False)

		# self.action = tf.placeholder(tf.float32, [None, 1], name='action')

		with tf.variable_scope('online_critic_1'):
			critic_1_fc1 = tf.layers.dense(self.batch_state, 256, tf.nn.relu, name='fc1')
			critic_1_fc2 = tf.layers.dense(tf.concat([critic_1_fc1, self.actor_out], axis=1), 128, tf.nn.relu, name='fc2')
			self.critic_1_out = tf.layers.dense(critic_1_fc2, 1, name='out')

		with tf.variable_scope('online_critic_2'):
			critic_2_fc1 = tf.layers.dense(self.batch_state, 256, tf.nn.relu, name='fc1')
			critic_2_fc2 = tf.layers.dense(tf.concat([critic_2_fc1, self.actor_out], axis=1), 128, tf.nn.relu, name='fc2')
			self.critic_2_out = tf.layers.dense(critic_2_fc2, 1, name='out')

		with tf.variable_scope('target_critic_1'):
			critic_target_1_fc1 = tf.layers.dense(self.batch_next_state, 256, tf.nn.relu, name='fc1', trainable=False)
			critic_target_1_fc2 = tf.layers.dense(tf.concat([critic_target_1_fc1, self.actor_target_out], axis=1), 128, tf.nn.relu, name='fc2', trainable=False)
			self.critic_1_target_out = tf.layers.dense(critic_target_1_fc2, 1, name='out', trainable=False)

		with tf.variable_scope('target_critic_2'):
			critic_target_2_fc1 = tf.layers.dense(self.batch_next_state, 256, tf.nn.relu, name='fc1', trainable=False)
			critic_target_2_fc2 = tf.layers.dense(tf.concat([critic_target_2_fc1, self.actor_target_out], axis=1), 128, tf.nn.relu, name='fc2', trainable=False)
			self.critic_2_target_out = tf.layers.dense(critic_target_2_fc2, 1, name='out', trainable=False)

		self.online_actor_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online_actor')
		self.target_actor_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor')
		self.online_critic_1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online_critic_1')
		self.online_critic_2_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online_critic_2')
		self.target_critic_1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic_1')
		self.target_critic_2_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic_2')

		with tf.variable_scope('actor_loss'):
			self.actor_loss = -1.0 * tf.reduce_mean(self.critic_1_out)
			self.actor_optim = tf.train.AdamOptimizer(1e-5).minimize(self.actor_loss, var_list=self.online_actor_params)

		self.batch_reward = tf.placeholder(tf.float32, [None, 1], name='batch_reward')

		# self.Q_target_for_loss = tf.placeholder(tf.float32, [None, 1], 'q_target')

		with tf.variable_scope('critic_loss'):
			critic_target_1 = self.critic_1_target_out
			critic_target_2 = self.critic_2_target_out
			Q = tf.minimum(critic_target_1, critic_target_2)
			self.Q_target = tf.stop_gradient(self.batch_reward + gamma * Q)

			self.critic_1_loss = tf.reduce_mean(tf.square(self.critic_1_out - self.Q_target))
			self.critic_2_loss = tf.reduce_mean(tf.square(self.critic_2_out - self.Q_target))

			self.critic_1_optim = tf.train.AdamOptimizer(3e-5).minimize(self.critic_1_loss, var_list=self.online_critic_1_params)
			self.critic_2_optim = tf.train.AdamOptimizer(3e-5).minimize(self.critic_2_loss, var_list=self.online_critic_2_params)

		

		# print(len(self.target_actor_params), len(self.target_critic_1_params), len(self.target_critic_2_params))

		with tf.variable_scope('target_hard_update'):
			self.target_hard_update = [tf.assign(t, e)
						for t, e in zip(self.target_actor_params + self.target_critic_1_params + self.target_critic_2_params, 
							self.online_actor_params + self.online_critic_1_params + self.online_critic_2_params)]


		with tf.variable_scope('target_soft_update'):
			self.tau = 0.005

			self.target_soft_update = [tf.assign(t, (1 - self.tau) * t + self.tau * e)
					for t, e in zip(self.target_actor_params + self.target_critic_1_params + self.target_critic_2_params, 
							self.online_actor_params + self.online_critic_1_params + self.online_critic_2_params)]

		with tf.variable_scope('model_save'):
			self.saver = tf.train.Saver(max_to_keep=1)

		self.initializer = tf.global_variables_initializer()

	def select_action(self, sess, state, epsilon):
		action = sess.run(self.actor_out, feed_dict={self.batch_state:state[np.newaxis, :]})
		action = np.squeeze(action)
		action = action + epsilon * np.random.normal(0, 0.2)
		action = 2 * np.clip(action, -1, 1)
		return float(action)


	def initial_network(self, sess):
		sess.run(self.initializer)


	def hard_update_target(self, sess):
		sess.run(self.target_hard_update)


	def soft_update_target(self, sess, tau=0.005):
		self.tau = tau
		sess.run(self.target_soft_update)


	def optimize_actor(self, sess, batch_state):
		_, loss = sess.run([self.actor_optim, self.actor_loss], feed_dict={self.batch_state:batch_state})
		return loss

	def optimize_critic(self, sess, batch_state, batch_next_state, batch_action, batch_reward):
		_, loss1, _, loss2 = sess.run([self.critic_1_optim, self.critic_1_loss, self.critic_2_optim, self.critic_2_loss], 
						feed_dict={self.actor_out:batch_action, self.batch_state:batch_state, self.batch_next_state:batch_next_state, self.batch_reward:batch_reward})
		return loss1, loss2

	def save_model(self, sess):
		self.saver.save(sess, 'td3_model/model.ckpt')

	def load_model(self, sess):
		self.saver.restore(sess, 'td3_model/model.ckpt')


if __name__ == '__main__':
	
	memory_replay = Memory(50000)

	ddpg = DDPG()

	writer = SummaryWriter('td3-logs')



	with tf.Session() as sess:
		sess.graph.finalize()
		ddpg.initial_network(sess)
		ddpg.hard_update_target(sess)
		# writer = tf.summary.FileWriter("graph/",sess.graph)
		# exit()
		env = gym.make('Pendulum-v0')
		for epoch in count():
			state = env.reset()
			episode_reward = 0
			for time_step in range(200):
				# start = time.time()
				action = ddpg.select_action(sess, state, epsilon)
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

					loss1, loss2 = ddpg.optimize_critic(sess, batch_state, batch_next_state, batch_action, batch_reward)
					writer.add_scalar('critic loss', loss1, learn_steps)
					writer.add_scalar('critic 2 loss', loss2, learn_steps)

					if learn_steps % 2 == 0:
						actor_loss = ddpg.optimize_actor(sess, batch_state)
						ddpg.soft_update_target(sess)
						# print('actor loss : ', actor_loss)
						writer.add_scalar('actor loss', actor_loss, learn_steps/2)
						

				if epsilon > 0:
					epsilon -= 1 / explore
				state = next_state
			writer.add_scalar('episode reward', episode_reward, epoch)
			if epoch % 1 == 0:
				print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))
			if (epoch + 1) % 10 == 0:
				ddpg.save_model(sess)
				print('model saved!')






