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
	max_episodes = 200000
	REPLAY_MEMORY = 400000
	env = Env()
	replay_buffer = deque()
	with tf.Session() as sess:
		c, r = env.pixel_size
		mainDQN = DQN(sess, (r, c, env.history_num), env.action_space, name="main")
		targetDQN = DQN(sess, (r, c, env.history_num), env.action_space, name="target")
		tf.global_variables_initializer().run()
		saver = tf.train.Saver()
		saver.restore(sess, "./ckpt/model.ckpt")

		copy_ops = m.get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
		sess.run(copy_ops)

		set_reward = 0
		total_reward = 0		
		reward = 0
		ready = False
		print("Set standard")
		while not ready:
			mask, img = env.preprocess_img(False)			
			ready = env.get_standard(mask)
		print("Ready")
		for episode in range(1, max_episodes + 1):		
			e = 1. / ((episode / 500) + 1)
			start1 = False
			start2 = False
			restart = False
			end = False
			history = env.history_reset()

			print("Start!")
			time.sleep(env.frame_time * 5)
			starttime = time.time()
			frame = 0
			try:
				while not end:
					if np.random.rand(1) < e:
						action = np.random.choice(range(env.action_space))
					else:
						action = np.argmax(mainDQN.predict(history))
					env.key_dict[action](env.frame_time)
					mask, img = env.preprocess_img()
					end = env.check_end(mask)
					if end:
						reward = env.reward
					else:		
						reward = 1
					next_history = env.history_update(history, img)
					replay_buffer.append((history, action, reward, next_history, end))
					if len(replay_buffer) > REPLAY_MEMORY:
					    replay_buffer.popleft()
					history = next_history
					total_reward += reward		
					time.sleep(env.frame_time - ((time.time() - starttime) % env.frame_time))	
					frame += 1
					# endtime = time.time()
					# print(endtime-starttime)
					# starttime = endtime
			except KeyboardInterrupt:
			    break					

			env.key_dict[1](env.frame_time)
			env.key_dict[3](env.frame_time)
			if reward > 0:
				print("Episode: {}, result: Win, reward:{}, frame:{}".format(episode, total_reward, frame))
			else:
				print("Episode: {}, result: Lose, reward:{}, frame:{}".format(episode, total_reward, frame))
			set_reward += total_reward
			total_reward = 0
			print("Not started yet")
			wait = 0
			while not start1 or start2 or not restart:
				start1 = start2
				mask, img = env.preprocess_img()
				start2 = env.check_start(mask)	
				restart = env.get_standard(mask, set_ = False)
				wait += 1
				if wait == 500:
					print("Game set: total reward: {}".format(set_reward))
					set_reward = 0
					saver.save(sess, "./ckpt/model.ckpt")			
					for _ in range(50):
					    minibatch = random.sample(replay_buffer, 100)
					    loss, _ = m.replay_train(mainDQN, targetDQN, minibatch)
					print("Loss: ", loss)
					print("Wait to restart to game")					
					for i in range(10):
						print("Let's start!")
						env.key_dict[5](env.frame_time)	
						time.sleep(env.frame_time)							
				if episode % mainDQN.update_target_rate == 1:
					sess.run(copy_ops)
			# time.sleep(env.frame_time * 5)

if __name__ == "__main__":

    main()
