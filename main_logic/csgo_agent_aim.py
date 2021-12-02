import ctypes
import cv2
import datetime
import keyboard
import mss
import numpy as np
import pyautogui
import random
import torch
import win32api
import win32gui
import win32ui
import win32con

from skimage.transform import resize
from time import sleep
from yolov5 import YOLOv5

# Models setting
yolov5_model_path = "yolov5/bestf.pt"
device = "cuda"

# Mouse movement setting
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0

# Initializes yolov5 model
yolov5 = YOLOv5(yolov5_model_path, device)

# Initializes screen capture library
sct = mss.mss()

# Gets the game window location and bring it to foreground.
ctypes.windll.shcore.SetProcessDpiAwareness(2)
hwnd = win32gui.FindWindow(None, "Counter-Strike: Global Offensive")
win32gui.SetForegroundWindow(hwnd)

# Sleeps for a while to wait for game frame to pop up.
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


# A sliding window that keeps the last X frames.
fired_last_frame = datetime.datetime.now()
play_as_CT = True

while True:
	# In real play, don't calculate gradient
	with torch.no_grad():
		start_time = datetime.datetime.now()
		# Captures the current game frame and converts into np array.
		img = np.array(sct.grab({"top": y, "left": x, "width": w, "height": h, "mon": -1}))
		# MSS stores raw pixels in the BGRA format, converts it to RGB.
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# Sets the yolo predict size to window width and make inference.
		prediction_results = yolov5.predict(img, size = w)


		# There should be one and only one image per Yolo prediction.
		# We can safely use 0 index to get result. Also copy from gpu to cpu for numpy to process.
		prediction_result = np.array(prediction_results.xywh[0].cpu())

		# Yolov5 output, first 4 numbers are bonding boxes
		# 5th number is probability
		# 6th number and after are the class labels
		pred_xywh = prediction_result[:, 0:4]
		pred_prob = prediction_result[:, 4]
		pred_class = prediction_result[:, 5:]


		# Gets the current cursor position
		mouse_position = pyautogui.position()
		# print('mouse_position', mouse_position)

		# Logic for moving cursor to target if detected
		top_targets = []
		second_targets = []
		mouse_moved = False
		for i in range(pred_class.size):
			#Search for CT.
			target_class = pred_class[i, 0]
			if play_as_CT:
				if (target_class == 2 and pred_prob[i] > 0.5):
					second_targets.append((x + pred_xywh[i,0], y + pred_xywh[i,1]))
				# Use different probability for different targets for accuracy
				elif (target_class == 3 and pred_prob[i] > 0.6):
					top_targets.append((x + pred_xywh[i,0], y + pred_xywh[i,1]))
			#else:			
				if (target_class == 0 and pred_prob[i] > 0.5):
					second_targets.append((x + pred_xywh[i,0], y + pred_xywh[i,1]))
				# Use different probability for different targets for accuracy
				elif (target_class == 1 and pred_prob[i] > 0.6):
					top_targets.append((x + pred_xywh[i,0], y + pred_xywh[i,1]))

		# Aiming priority (aiming at enemy head > enemy body)
		if len(top_targets) > 0 or len(second_targets) > 0:
			mouse_moved = True

			final_target_x = 0;
			final_target_y = 0;
			if len(top_targets) > 0:
			#if False:
				top_targets = sorted(top_targets, key=lambda x: abs(mouse_position.x - x[0]) + abs(mouse_position.y - x[1]))
				final_target_x = top_targets[0][0]
				final_target_y = top_targets[0][1] + 2
			elif len(second_targets) > 0:
				second_targets = sorted(second_targets, key=lambda x: abs(mouse_position.x - x[0]) + abs(mouse_position.y - x[1]))
				final_target_x = second_targets[0][0]
				final_target_y = second_targets[0][1] - 5
			# Move the cursor to the target
			move_x = final_target_x - mouse_position.x
			move_y = final_target_y - mouse_position.y

			pyautogui.move(move_x * 1.3, move_y * 1.3)
			# Fire when cursor is within 10 pixel to target
			if abs(mouse_position.x - final_target_x) + abs(mouse_position.y - final_target_y) < 25:
				fired_duration = datetime.datetime.now() - fired_last_frame
				if fired_duration.total_seconds() > 0.2:
					pyautogui.click()
					fired_last_frame = datetime.datetime.now()

		duration = datetime.datetime.now() - start_time
		print('FPS ', 1 / duration.total_seconds())
		if keyboard.is_pressed('f1'):
			break;
