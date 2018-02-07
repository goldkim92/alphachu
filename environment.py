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
		self.observation_space = (283, 430)
		self.history_num = 6
		self.key_dict = {0: c.left_s,
						1: c.left_e,
			            2: c.right_s,
			            3: c.right_e,
			            4: c.up,
			            5: c.p,
			            6: c.p_down}

		self.action_space = len(self.key_dict)
		self.win_reward = 500
		self.lose_penalty = -500

		self.frame_time = 1/15
		self.sct = mss()
		self.sct.compression_level = 9		
		self.screen_size = (100, 100, 570, 430)  # top left width height
		# self.screen_size = (300, 600, 820, 980)  # top left width height
		self.pixel_size = (105, 70)
		self.lower_red = np.array([0, 200, 120])
		self.upper_red = np.array([10, 255, 150])
		self.lower_yellow = np.array([20, 120, 100])
		self.upper_yellow = np.array([30, 255, 255])
		# self.lower_red = np.array([20, 120, 100])
		# self.upper_red = np.array([30, 130, 255])

		self.pole = 0
		self.ground = 0
		self.reward = None

		self.threshold = 0.95
		self.set_templates()

	def set_templates(self):
		self.polepoint = cv2.imread('img/polepoint/polepoint.png',0)
		methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
		self.template_method = eval(methods[1])

	def preprocess_img(self, resize = True):
		img = self.sct.grab(self.screen_size)
		img = np.array(img)
		if resize:
			# r, c = img.shape
			img = img[self.ground - 183: self.ground + 100, self.pole - 215:self.pole + 215]
		img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		img_red = cv2.inRange(img_hsv, self.lower_red, self.upper_red)
		img_yellow = cv2.inRange(img_hsv, self.lower_yellow, self.upper_yellow)
		mask = img_red + img_yellow
		img = cv2.resize(img, self.pixel_size)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# output = cv2.bitwise_and(img_gray, img_gray, mask = mask)
		# cv2.circle(mask,(int(self.pole),int(self.ground)), 5, (150,200,200), -1)
		cv2.imshow('img', img_gray)
		cv2.waitKey(1)		
		return mask, img_gray

	def history_reset(self):
		c, r = self.pixel_size
		state = np.zeros((r, c))
		history = np.stack((state, state, state, state, state, state), axis=2)
		history = np.reshape([history], (1, r, c, self.history_num))
		return history

	def history_update(self, history, state):
		c, r = self.pixel_size
		state = np.reshape([state], (1, r, c, 1))
		next_history = np.append(state, history[:, :, :, :self.history_num - 1], axis=3)
		return next_history

	def get_standard(self, img, set_ = True):
		gameset_match = cv2.matchTemplate(img, self.polepoint, eval('cv2.TM_CCOEFF_NORMED'))
		if np.max(gameset_match) > self.threshold:
			if set_:	
				min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gameset_match)
				top_left = max_loc
				w, h = self.polepoint.shape[::-1]
				x, y = (top_left[0] + w//2, top_left[1] + h//2)	
				self.pole = x
				self.ground = y	
			return True
		else:
			return False

	def check_start(self, img):	
		return np.sum(img[-10:]) == 0

	def check_end(self, img):
		if np.sum(img[-2:,:]) > 1000:
			if np.sum(img[-2:, :215]) > np.sum(img[-2:, 215:]):
				self.reward = self.win_reward
				return True
			else:
				self.reward = self.lose_penalty
				return True
		else:
			return False
