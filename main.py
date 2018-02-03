# from PIL import Image
# from mss import mss
# from pynput.keyboard import Key, Controller
import numpy as np
import tensorflow as tf
import random
from collections import deque
import time

import model as m
from model import DQN
import environment as e
from environment import Env

def main():
	max_episodes = 5000
	REPLAY_MEMORY = 10000
	env = Env()
	replay_buffer = deque()
	with tf.Session() as sess:
		mainDQN = DQN(sess, env.observation_space * env.history_num, env.action_space, name="main")
		targetDQN = DQN(sess, env.observation_space * env.history_num, env.action_space, name="target")
		tf.global_variables_initializer().run()
		saver = tf.train.Saver()
		saver.restore(sess, "./ckpt/model.ckpt")

		copy_ops = m.get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
		sess.run(copy_ops)

		set_reward = 0
		total_reward = 0		
		reward = 0
		for episode in range(1, max_episodes + 1):
			env.key_dict[0](env.frame_time)

			if reward <= 0:
				ball_side = 'left'
			else:
				ball_side = 'right'
			e = 1. / ((episode / 200) + 1)
			# e = 1 / 3
			start = False
			restart = False
			end = False
			history = env.history_reset()
			print("Not started yet")
			wait = 0
			while not start:
				img = env.preprocess_img()
				start = env.check_start(img)
				wait += 1
				if wait == 1000:
					for i in range(10):
						print("Let's start!")
						# env.key_dict[5](env.frame_time)						
						env.key_dict[6](env.frame_time)								
						env.key_dict[0](env.frame_time)						
			for i in range(20):
				env.preprocess_img()
			img = env.preprocess_img()
			env.get_standard(img)
			print("Start!")
			prev_state = np.zeros(env.observation_space)
			frame = 0
			try:
				while not end:
					frame += 1
					if np.random.rand(1) < e:
						action = np.random.choice(range(9))
					else:
						action = np.argmax(mainDQN.predict(history))
					env.key_dict[action](env.frame_time)
					img = env.preprocess_img()
					state, side = env.get_state(img)
					if side != 'unclear' and state[1] < 30:
						ball_side = side
					end, game_set, reward = env.check_end(img, ball_side)
					if not end:
						reward = 1			
					# if ((state != prev_state).any()) and (side != 'unclear'):
					next_history = env.history_update(history, state)
					replay_buffer.append((history, action, reward, next_history, end))
					if len(replay_buffer) > REPLAY_MEMORY:
					    replay_buffer.popleft()
					history = next_history
						# print(next_history)
					prev_state = state
					total_reward += reward					
			except KeyboardInterrupt:
			    break										

			if reward > 0:
				print("Episode: {}, result: Win, reward:{}, frame:{}".format(episode, total_reward, frame))
			else:
				print("Episode: {}, result: Lose, reward:{}, frame:{}".format(episode, total_reward, frame))
			set_reward += total_reward
			total_reward = 0
			if game_set:
				env.key_dict[0](env.frame_time)								
				print("Game set: total reward: {}".format(set_reward))
				set_reward = 0
				saver.save(sess, "./ckpt/model.ckpt")			
				for _ in range(100):
				    minibatch = random.sample(replay_buffer, 100)
				    loss, _ = m.replay_train(mainDQN, targetDQN, minibatch)
				print("Loss: ", loss)
				sess.run(copy_ops)
				print("Wait to restart to game")
				while not restart:
					img = env.preprocess_img()
					restart = env.check_restart(img)
				print("restart")
				for i in range(10):
					# env.key_dict[5](env.frame_time)								
					env.key_dict[6](env.frame_time)	

					
					env.key_dict[0](env.frame_time)								



	# bot_play(mainDQN)

        # env2 = wrapper.Monitor(env, 'gym-results', force=True)

        # for i in range(200):

        #     bot_play(mainDQN, env=env2)

        # env2.colse()

if __name__ == "__main__":

    main()
