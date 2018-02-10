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
	max_episodes = 40000
	REPLAY_MEMORY = 400000
	env = Env(max_episodes)
	replay_buffer = deque()
	with tf.Session() as sess:
		c, r = env.pixel_size
		mainDQN = DQN(sess, (r, c, env.history_num), env.action_space, name="main")
		targetDQN = DQN(sess, (r, c, env.history_num), env.action_space, name="target")
		tf.global_variables_initializer().run()
		saver = tf.train.Saver()
		saver.restore(sess, "./ckpt/model.ckpt")
		writer = tf.summary.FileWriter("summary/dqn", sess.graph)
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
					predicts = mainDQN.predict(history)
					env.avg_q_max += np.max(predicts)
					if np.random.rand(1) < env.epsilon:
						action = np.random.choice(range(env.action_space))
					else:
						action = np.argmax(predicts)
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
					if env.epsilon > env.epsilon_end:
						env.epsilon -= env.epsilon_decay_step
					# endtime = time.time()
					# print(endtime-starttime)
					# starttime = endtime
			except KeyboardInterrupt:
			    break					

			stats = [total_reward, env.avg_q_max / float(frame), frame, env.avg_loss / float(frame)]
			for i in range(len(stats)):
				sess.run(env.update_ops[i], feed_dict={env.summary_placeholders[i]: float(stats[i])})
			summary_str = sess.run(env.summary_op)
			writer.add_summary(summary_str, episode + 1)
			env.avg_q_max, env.avg_loss = 0, 0

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
					for _ in range(50):
					    minibatch = random.sample(replay_buffer, 100)
					    loss, _ = m.replay_train(mainDQN, targetDQN, minibatch)
					    env.avg_loss += loss
					saver.save(sess, "./ckpt/model.ckpt")			
					print("Loss: ", loss)
					print("Wait to restart to game")						
					env.key_dict[6](env.frame_time)	
					time.sleep(env.frame_time)		
				if wait > 2000:					
					print("Let's start!")
					env.key_dict[6](env.frame_time)	
					time.sleep(env.frame_time)		

			if episode % mainDQN.update_target_rate == 1:
				sess.run(copy_ops)
			# time.sleep(env.frame_time * 5)

if __name__ == "__main__":

    main()
