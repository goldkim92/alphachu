import cv2
from mss import mss
from pynput.keyboard import Key, Controller
import numpy as np
import time

import control as c

import numpy as np
import tensorflow as tf
import random
from collections import deque

class Env:
	def __init__(self):
		self.observation_space = 7
		self.history_num = 12
		self.key_dict = {0: c.stay,
						1: c.continu,
			            2: c.left,
			            3: c.right,
			            4: c.up,
			            5: c.down,
			            6: c.p,
			            7: c.p_left,
			            8: c.p_right,
			            9: c.p_up,
			            10: c.p_down}
		self.action_space = len(self.key_dict)
		self.win_reward = 100
		self.lose_penalty = -100

		self.frame_time = 1/30
		self.sct = mss()
		self.sct.compression_level = 9		
		self.screen_size = (100, 100, 520, 380)  # top left width height
		# self.screen_size = (300, 600, 720, 880)  # top left width height
		self.pixel_size = (420, 280)
		self.lower_red = np.array([0, 200, 60])
		self.upper_red = np.array([30, 255, 255])

		self.pole = None
		self.ground = None

		self.threshold = 0.8
		self.set_templates()

	def set_templates(self):
		self.start = cv2.imread('img/start/start.png',0)
		self.restart = cv2.imread('img/restart/restart.png',0)
		self.ready = cv2.imread('img/ready/ready.png',0)
		self.gameset = cv2.imread('img/gameset/gameset.png',0)
		self.balls = self.prepare_templates('ball', 5)
		self.left_eye = self.prepare_templates('eyel', 7)
		self.right_eye = self.prepare_templates('eyer', 7)
		methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
		self.template_method = eval(methods[1])

	def prepare_templates(self, pic, num):
		templates = []
		for i in range(1, num + 1):
			img_name = 'img/' + pic + '/' + pic + str(i) + '.png'
			template = cv2.imread(img_name,0)
			templates.append(template)
		return templates

	def preprocess_img(self):
		img = self.sct.grab(self.screen_size)
		img = np.array(img)
		# img = cv2.resize(img, self.pixel_size)
		img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		img = cv2.inRange(img_hsv, self.lower_red, self.upper_red)
		cv2.imshow('img', img)
		cv2.waitKey(1)		
		return img

	def history_reset(self):
		one_state = np.zeros(self.observation_space)
		history = np.array([[one_state] * self.history_num])[0].flatten()
		return history

	def history_update(self, history, state):
		history = history[:(self.history_num -1)* self.observation_space]
		next_history = np.append(state, history)
		return next_history

	def get_standard(self, img, region = 0.4):
	    height, width = img.shape
	    start = int(width * region)
	    end = int(width * (1-region))
	    self.pole = start + np.argmax(np.sum(img[: , start:end], axis =0))
	    self.ground = np.argmax(np.sum(img, axis =1))

	def check_start(self, img):
		start_match = cv2.matchTemplate(img, self.start, self.template_method)
		ready_match = cv2.matchTemplate(img, self.ready, self.template_method)
		if (np.max(start_match) > self.threshold) or (np.max(ready_match) > self.threshold):
			if (np.max(ready_match) > self.threshold):
				time.sleep(1.3)
			return True
		else:
			return False

	def check_restart(self, img):
		restart_match = cv2.matchTemplate(img, self.restart, self.template_method)
		if np.max(restart_match) > self.threshold:
			return True
		else:
			return False

	def max_estimate(self, img, templates):
		max_tem = None
		max_res = None
		max_num = 0
		for template in templates:
			res = cv2.matchTemplate(img, template, self.template_method)	
			if np.max(res) > max_num:
				max_num = np.max(res)
				max_tem = template
				max_res = res
		if max_num != 0:
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(max_res)
			top_left = max_loc
			w, h = max_tem.shape[::-1]
			x, y = (top_left[0] + w//2, top_left[1] + h//2)			
			return (x, y, max_num * 100)
		else:
			return (0, 0, 0)


	def get_state(self, img):
		state = np.zeros(self.observation_space)
		state[:3] = self.max_estimate(img, self.balls)
		x, y, z = self.max_estimate(img[50:, :self.pole], self.left_eye)
		state[3], state[4] = x, y 
		x, y, z = self.max_estimate(img[50:, self.pole:], self.right_eye)
		state[5], state[6] = x, y
		state[4] += 50
		state[6] += 50
		state[5] += self.pole
		# img = cv2.circle(img,(self.pole,self.ground), 3, (200,200,200), -1)
		# img = cv2.circle(img,(int(state[0]),int(state[1])), 3, (100,100,100), -1)
		# img = cv2.circle(img,(int(state[3]),int(state[4])), 3, (50,50,50), -1)
		# img = cv2.circle(img,(int(state[6]),int(state[7])), 3, (150,150,150), -1)
		# if (state[2] > self.threshold * 100):
		# 	img = cv2.line(img,(self.pole,self.ground),(int(state[0]),int(state[1])), (100,100,100), 2)
		# if (state[5] > self.threshold * 100):
		# 	img = cv2.line(img,(self.pole,self.ground),(int(state[3]),int(state[4])), (50,50,50), 2)
		# if (state[8] > self.threshold * 100):
		# 	img = cv2.line(img,(self.pole,self.ground),(int(state[6]),int(state[7])), (150,150,150), 2)

		# cv2.imshow('img', img)
		# cv2.waitKey(1)		
		state[0] = state[0] - self.pole
		state[3] = state[3] - self.pole
		state[5] = state[5] - self.pole
		state[1] = self.ground - state[1]
		state[4] = self.ground - state[4]
		state[6] = self.ground - state[6]

		if (state[2] > self.threshold * 100):
			if state[0] < 0:
				return state, 'left'
			else:
				return state, 'right'
		else:
			return state, 'unclear'

	def check_end(self, img, side):
		gameset_match = cv2.matchTemplate(img, self.gameset, self.template_method)
		# img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
		# print(np.sum(img))
		game_set = (np.max(gameset_match) > self.threshold)
		dark = (np.sum(img) < 1000000)
		if game_set or dark:
			if side == 'left':
	   			return True, game_set, self.win_reward
			else:
				return True, game_set, self.lose_penalty
		else:
			return False, False, 0