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

lower_red = np.array([0, 200, 60])  # example value
upper_red = np.array([30, 255, 255])  # example value

sct = mss()
sct.compression_level = 9
frame_num = 0

img = sct.grab(screen_size)
img = np.array(img)
(img[0], img[2]) = (img[2], img[0])  # to rgb
img = cv2.resize(img, pixel_size)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = cv2.inRange(img_hsv, lower_red, upper_red)
r = cv2.selectROI(img, False)
roi = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
cv2.imwrite('gameset.png', roi)