# csgo-agent-cs230

This project builds a deep learning CS:GO AI agent (cs230 final project).
It relies on YOLOv5 model for enemy detection and a custom sequence model for
movement and orientation prediction.

Folders:

main_logic: Contains logic to interact with the game (aim mode and deathmatch mode) and custom RNN movement model.

data_capture: Contains various scripts to capture training data

data_engineering: Contains scripts and data examples to investigate YOLO layers

yolov5: I modified yolov5 open source code for this project purpose. Specifically,
	I modified the plots.py and yolo.py to output intermediate Conv layer images
	for movement network traning.

Open source project used:

yolov5 git repository: https://github.com/ultralytics/yolov5

MSS git repository: https://github.com/BoboTiG/python-mss

Pygame git repository: https://github.com/pygame/pygame

Python keyboard git repository https://github.com/boppreh/keyboard

VGgamepad git repository: https://github.com/PJSoftCo/VGamepad

