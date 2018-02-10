from matplotlib import pyplot as plt
import cv2
# from PIL import ImageGrab
from PIL import Image
from mss import mss
from pynput.keyboard import Key, Controller
import numpy as np
import sched
import time
import control as c

frame_time = 1/15

screen_size = (100, 100, 520, 380)  # top left width height
pixel_size = (420, 280)

# lower_red = np.array([0, 200, 60])  # example value
# upper_red = np.array([30, 255, 255])  # example value
lower_red = np.array([0, 200, 120])
upper_red = np.array([10, 255, 150])
lower_yellow = np.array([20, 120, 100])
upper_yellow = np.array([30, 255, 255])
sct = mss()
sct.compression_level = 9
frame_num = 0
splash = cv2.imread('img/polepoint/polepoint.png',0)
while True:
	img = sct.grab(screen_size)
	img = np.array(img)
	(img[0], img[2]) = (img[2], img[0])  # to rgb
	img = cv2.resize(img, pixel_size)
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img_red = cv2.inRange(img_hsv, lower_red, upper_red)
	img_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
	mask = img_red + img_yellow
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	output = cv2.bitwise_and(img_gray, img_gray, mask = mask)
	gameset_match = cv2.matchTemplate(output, splash, eval('cv2.TM_CCOEFF_NORMED'))
		# # if np.max(gameset_match) > 0.7:	
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gameset_match)
	top_left = max_loc
	w, h = splash.shape[::-1]
	x, y = (top_left[0] + w//2, top_left[1] + h//2)		
	print(x, y)		
	# region = 0.4
	# height, width = output.shape
	# start = int(width * region)
	# end = int(width * (1-region))
	# pole = start + np.argmax(np.sum(output[: , start:end], axis =0))
	# ground = np.argmax(np.sum(output, axis =1))	
	# print(pole, ground)
	cv2.circle(output,(x,y), 5, (150,200,200), -1)
	# print(np.max(gameset_match))
	cv2.imshow('img', output)
	cv2.waitKey(1)		
# r = cv2.selectROI(mask, False)
# roi = output[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
# cv2.imwrite('splash.png', roi)