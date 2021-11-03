import ctypes
import cv2
import datetime
import keyboard
import matplotlib.pyplot as plt
import mss
import numpy as np
import pyautogui
import pygame
import win32gui
import win32ui
import win32con

from skimage.transform import resize
from time import sleep
from yolov5 import YOLOv5

# This script is used for capturing real player actions and mouse
# orientation per frame during a real game play.

pygame.init()
controller = pygame.joystick.Joystick(0)
controller.init()

model_path = "yolov5/bestf.pt"
device = "cuda"

# Mouse movement setting
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0

# Opens file to record data
f1 = open('data2/train_s1.txt','a')
f2 = open('data2/train_s5.txt','a')
f3 = open('data2/label.txt','a')

# init yolov5 model
yolov5 = YOLOv5(model_path, device)
sct = mss.mss()

# Gets the game window location
ctypes.windll.shcore.SetProcessDpiAwareness(2)
hwnd = win32gui.FindWindow(None, "Counter-Strike: Global Offensive")
win32gui.SetForegroundWindow(hwnd)

# Sleeps for 1 sec to wait for game frame to pop up.
sleep(1)

# Gets the game window location
rect = win32gui.GetWindowRect(hwnd)
x = rect[0]
y = rect[1]
w = rect[2] - x
h = rect[3] - y
print("Window captured %s:" % win32gui.GetWindowText(hwnd))
print("Location: (%d, %d)" % (x, y))
print("Size: (%d, %d)" % (w, h))

# Sleeps for 1 sec to wait for game to center the mouse.
sleep(1)

# Gets the cneter coordinate of mouse
# center = pyautogui.position()
# print("Mouse center is ", center)
# sleep(1)

i = 0
while True:
	start = datetime.datetime.now()
	if i % 100 == 0:
		print(i)
	i = i + 1

	# Gets the current game frame into np array.
	img = np.array(sct.grab({"top": y, "left": x, "width": w, "height": h, "mon": -1}))

	# MSS stores raw pixels in the BGRA format, converts it to RGB.
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# time1 = datetime.datetime.now()
	# screen_shot_time = time1 - start
	# print('screen_shot_time ', screen_shot_time.total_seconds())

	keys = np.zeros(9)
	# Uncomment the following section if using mouse coordinates as training data
	# current_position = pyautogui.position()
	# keys[7] = current_position.x - center.x
	# keys[8] = current_position.y - center.y
	pygame.event.pump()

	# Records the joystick direction for this frame
	keys[7] = controller.get_axis(2)
	keys[8] = controller.get_axis(3)

	# Records which key is pressed for this frame
	if keyboard.is_pressed('w'):
		keys[0] = 1
	if keyboard.is_pressed('s'):
		keys[1] = 1
	if keyboard.is_pressed('a'):
		keys[2] = 1
	if keyboard.is_pressed('d'):
		keys[3] = 1
	if keyboard.is_pressed(' '):
		keys[4] = 1
	if keyboard.is_pressed('r'):
		keys[5] = 1
	if keyboard.is_pressed('ctrl'):
		keys[6] = 1
	# time2 = datetime.datetime.now()
	# keyboard_time = time2 - time1
	# print('keyboard_time ', keyboard_time.total_seconds())

	# Sets the yolo predict size to window width.
	prediction_results = yolov5.predict(img, size = w)

	# time3 = datetime.datetime.now()
	# predict_time = time3 - time2
	# print('predict_time ', predict_time.total_seconds())

	# This is a hack. I'm changing the underlying yolov5 library to output a global variable
	# that contains the intermediate layer image feature. This avoids using another
	# network to do image feature extraction since yolov5 is already doing it.
	from yolov5.utils.plots import image_feature_s1
	from yolov5.utils.plots import image_feature_s5

	image_s1 = image_feature_s1.cpu().detach().float().numpy()
	image_s5 = image_feature_s5.cpu().detach().numpy()
	# width = image.shape[1]
	# height = image.shape[0]
	# print("image w x h ", width, height)

	# These magic numbers are chosen carefully.
	# It converts the feature image from 136x216 to 90x170 to reduce the
	# input size while keeping most of the useful information. 

	image_s1 = image_s1[28:118,23:193]
	# image s1 will go through a compress phase because the origional is too big
	image_s1 = resize(image_s1, (27, 44))
	print("image_s1 size", image_s1.shape)

	image_s5 = image_s5[4:31,5:49]
	print("image_s5 size", image_s5.shape)

	# Save a feature image to manually exam.
	# plt.imshow(image_s1)

	feature_vector_s1 = image_s1.reshape(1, -1)
	feature_vector_s5 = image_s5.reshape(1, -1)
	# print("feature_vector shape: ", feature_vector.shape)

	keys = keys.reshape(1, 9)
	# print("keys_vector shape: ", keys.shape)

	np.savetxt(f1, feature_vector_s1, delimiter=',', fmt='%.5f')
	np.savetxt(f2, feature_vector_s5, delimiter=',', fmt='%.5f')
	np.savetxt(f3, keys, delimiter=',', fmt='%.5f')
	# time4 = datetime.datetime.now()
	# write_time = time4 - time3
	# print('write_time ', write_time.total_seconds())


################### Uncomment the following section if turn on auto aim when collecting data ############

	# There should be one and only one image per prediction.
	# We can safely use 0 index to get result. Also copy from gpu to cpu for numpy to process.
	prediction_result = np.array(prediction_results.xywh[0].cpu())

	# Yolov5 output, first 4 digits are bonding boxes
	# 5th digit is probability
	# 6th digit and after are the class labels
	pred_xywh = prediction_result[:, 0:4]
	pred_prob = prediction_result[:, 4]
	pred_class = prediction_result[:, 5:]

	mouse_position = pyautogui.position()
	# print('mouse_position', mouse_position)

	# Logic for moving mouse to target if detected
	top_targets = []
	second_targets = []
	for i in range(pred_class.size):
		#Search for CT.
		target_class = pred_class[i, 0]
		if (target_class == 0 and pred_prob[i] > 0.4):
			second_targets.append((x + pred_xywh[i,0], y + pred_xywh[i,1]))
		# Use different probability for different targets for accuracy
		elif (target_class == 1 and pred_prob[i] > 0.5):
			top_targets.append((x + pred_xywh[i,0], y + pred_xywh[i,1]))

	if len(top_targets) > 0 or len(second_targets) > 0:
		final_target_x = 0;
		final_target_y = 0;
		if len(top_targets) > 0:
			top_targets = sorted(top_targets, key=lambda x: abs(mouse_position.x - x[0]) + abs(mouse_position.y - x[1]))
			final_target_x = top_targets[0][0]
			final_target_y = top_targets[0][1] + 5
		elif len(second_targets) > 0:
			second_targets = sorted(second_targets, key=lambda x: abs(mouse_position.x - x[0]) + abs(mouse_position.y - x[1]))
			final_target_x = second_targets[0][0]
			final_target_y = second_targets[0][1]
		# moveTo is not accurate, change to move later
		pyautogui.move((final_target_x - mouse_position.x), (final_target_y - mouse_position.y))
		# Fire when we are with in 10 pixel to target
		if abs(mouse_position.x - final_target_x) + abs(mouse_position.y - final_target_y) < 10:
			pyautogui.click()


#########################################################################

	duration = datetime.datetime.now() - start
	print('total time spent ', duration.total_seconds())
	if keyboard.is_pressed('F1'):
		break