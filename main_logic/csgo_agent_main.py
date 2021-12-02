import ctypes
import cv2
import datetime
import keyboard
import mss
import numpy as np
import pyautogui
import random
import torch
import vgamepad as vg
import win32api
import win32gui
import win32ui
import win32con

from movement_network import MovementNetwork
from skimage.transform import resize
from time import sleep
from yolov5 import YOLOv5

# Models setting
yolov5_model_path = "yolov5/bestf.pt"
movement_model_path = "movement_network_60s_5l_6.pt"
device = "cuda"

# Mouse movement setting
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0

# Wether to use virtual gamepad or mouse to control orientation.
gamepad_mode = True

# Number of frames to use as one sequence for movement prediction
sequence_length = 60

# Initializes yolov5 model
yolov5 = YOLOv5(yolov5_model_path, device)

# Initializes movement network model and push to GPU
movement_network = MovementNetwork().cuda()
movement_network.load_state_dict(torch.load(movement_model_path), strict = True)
movement_network.eval()

# Initializes screen capture library
sct = mss.mss()

# Gets the game window location and bring it to foreground.
ctypes.windll.shcore.SetProcessDpiAwareness(2)
hwnd = win32gui.FindWindow(None, "Counter-Strike: Global Offensive")
win32gui.SetForegroundWindow(hwnd)

# Sleeps for a while to wait for game frame to pop up.
sleep(2)

# Gets the game window location
rect = win32gui.GetWindowRect(hwnd)
x = rect[0]
y = rect[1]
w = rect[2] - x
h = rect[3] - y
print("Window captured %s:" % win32gui.GetWindowText(hwnd))
print("Location: (%d, %d)" % (x, y))
print("Size: (%d, %d)" % (w, h))

# Initializes virtual gamepad
gamepad = vg.VDS4Gamepad()
gamepad.right_joystick_float(x_value_float = 0.0, y_value_float=0.0)
gamepad.update()
sleep(2)

# A sliding window that keeps the last X frames.
frame_window = []
fired_last_frame = datetime.datetime.now()
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

		# The following is a hack. I changed the underlying yolov5 library to output a global
		# variable that contains the intermediate Conv layer image feature. This avoids using another
		# network to do image feature extraction since yolov5 is already doing it.

		from yolov5.utils.plots import image_feature_s1
		# Corp and reshape yolo image feature to expected size
		image_feature_s1 = image_feature_s1[28:118,23:193].float().detach().cpu().numpy()

		# image s1 will go through a downsampling process because the origional is still too big
		image_feature_s1 = resize(image_feature_s1, (27, 44))
		image_vector = torch.from_numpy(image_feature_s1).reshape(1,-1).to(device)

		# from yolov5.utils.plots import image_feature_s5
		# image_vector = image_feature_s5[4:31,5:49].reshape(1,-1).float()

		# Adds the current game frame into sliding window
		frame_window.append(image_vector)

		# Only keeps the last sequence_length frames in sliding window
		if len(frame_window) > sequence_length:
			frame_window.pop(0)
		image_sequence = torch.stack(frame_window, dim=1)

		# Predicts movement
		keyboard_prediction, mouse_prediction = movement_network(image_sequence)
		keyboard_prediction = torch.sigmoid(keyboard_prediction)

		# There should be one and only one image per Yolo prediction.
		# We can safely use 0 index to get result. Also copy from gpu to cpu for numpy to process.
		prediction_result = np.array(prediction_results.xywh[0].cpu())

		# Yolov5 output, first 4 numbers are bonding boxes
		# 5th number is probability
		# 6th number and after are the class labels
		pred_xywh = prediction_result[:, 0:4]
		pred_prob = prediction_result[:, 4]
		pred_class = prediction_result[:, 5:]

		# print('pred_xywh', pred_xywh)
		# print('pred_prob', pred_prob)
		# print('pred_class', pred_class)

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
			if (target_class == 0 and pred_prob[i] > 0.55):
				second_targets.append((x + pred_xywh[i,0], y + pred_xywh[i,1]))
			# Use different probability for different targets for accuracy
			elif (target_class == 1 and pred_prob[i] > 0.6):
				top_targets.append((x + pred_xywh[i,0], y + pred_xywh[i,1]))

		# Aiming priority (aiming at enemy head > enemy body)
		if len(top_targets) > 0 or len(second_targets) > 0:
			mouse_moved = True
			# Disable gamepad input if an enemy is detected
			gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
			gamepad.update()

			final_target_x = 0;
			final_target_y = 0;
			if len(top_targets) > 0:
				top_targets = sorted(top_targets, key=lambda x: abs(mouse_position.x - x[0]) + abs(mouse_position.y - x[1]))
				final_target_x = top_targets[0][0]
				final_target_y = top_targets[0][1] + 2
			elif len(second_targets) > 0:
				second_targets = sorted(second_targets, key=lambda x: abs(mouse_position.x - x[0]) + abs(mouse_position.y - x[1]))
				final_target_x = second_targets[0][0]
				final_target_y = second_targets[0][1]
			# Move the cursor to the target
			pyautogui.move((final_target_x - mouse_position.x) * 1.1, (final_target_y - mouse_position.y) * 1.1)
			# Fire when cursor is within 10 pixel to target
			if abs(mouse_position.x - final_target_x) + abs(mouse_position.y - final_target_y) < 20:
				fired_duration = datetime.datetime.now() - fired_last_frame
				if fired_duration.total_seconds() > 0.25:
					pyautogui.click()
					fired_last_frame = datetime.datetime.now()

		# Logic for mouse orientation if no target is detected.
		# print("mouse", mouse_prediction)
		# print("move_mouse", mouse_prediction)
		if (not gamepad_mode) and (not mouse_moved):
			pyautogui.move(mouse_prediction[0,0] * 4, mouse_prediction[0,1] * 4)
		if gamepad_mode and (not mouse_moved):
			x_value = float(mouse_prediction[0,0])
			y_value = float(mouse_prediction[0,1])
			gamepad.right_joystick_float(x_value_float=x_value, y_value_float=y_value)  # value between 0 and 255
			gamepad.update()


		# Apply keyboard predicitons
		if keyboard_prediction[0,0] >= 0.5:
			win32api.keybd_event(0x57, win32api.MapVirtualKey(0x57, 0), 0, 0)  # press W
		else:
			win32api.keybd_event(0x57, win32api.MapVirtualKey(0x57, 0), win32con.KEYEVENTF_KEYUP, 0)  # lift W

		if keyboard_prediction[0,1] >= 0.5:
			win32api.keybd_event(0x53, win32api.MapVirtualKey(0x53, 0), 0, 0)  # press S
		else:
			win32api.keybd_event(0x53, win32api.MapVirtualKey(0x53, 0), win32con.KEYEVENTF_KEYUP, 0)  # lift S

		if keyboard_prediction[0,2] >= 0.5:
			win32api.keybd_event(0x41, win32api.MapVirtualKey(0x41, 0), 0, 0)  # press A
		else:
			win32api.keybd_event(0x41, win32api.MapVirtualKey(0x41, 0), win32con.KEYEVENTF_KEYUP, 0)  # lift A

		if keyboard_prediction[0,3] >= 0.5:
			win32api.keybd_event(0x44, win32api.MapVirtualKey(0x44, 0), 0, 0)  # press D
		else:
			win32api.keybd_event(0x44, win32api.MapVirtualKey(0x44, 0), win32con.KEYEVENTF_KEYUP, 0)  # lift D

		if keyboard_prediction[0,4] >= 0.5:
			win32api.keybd_event(0x20, win32api.MapVirtualKey(0x20, 0), 0, 0)  # press space
		else:
			win32api.keybd_event(0x20, win32api.MapVirtualKey(0x20, 0), win32con.KEYEVENTF_KEYUP, 0)  # lift space

		if keyboard_prediction[0,5] >= 0.5:
			win32api.keybd_event(0x52, win32api.MapVirtualKey(0x52, 0), 0, 0)  # press R
		else:
			win32api.keybd_event(0x52, win32api.MapVirtualKey(0x52, 0), win32con.KEYEVENTF_KEYUP, 0)  # lift R

		if keyboard_prediction[0,6] >= 0.5:
			win32api.keybd_event(0x11, win32api.MapVirtualKey(0x11, 0), 0, 0)  # press Ctrl
		else:
			win32api.keybd_event(0x11, win32api.MapVirtualKey(0x11, 0), win32con.KEYEVENTF_KEYUP, 0)  # lift Ctrl

		duration = datetime.datetime.now() - start_time
		print('FPS ', 1 / duration.total_seconds())
		if keyboard.is_pressed('f1'):
			break;



# Process yolov5 object detection output
def ProcessYoloPredictionResult(prediction_result, drop_threshold):
	pred_xywh = prediction_result[:, 0:4]
	pred_conf = prediction_result[:, 4]
	pred_prob = prediction_result[:, 5:]
	pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5, 
		pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
