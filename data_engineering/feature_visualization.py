from matplotlib import pyplot as plt
from movement_network import MovementNetwork
from yolov5 import YOLOv5
import numpy as np
import torch

# This script is used for visulaizing intermediate Conv layers image
# features in YOLO model.

model_path = "yolov5/bestf.pt"
device = "cuda"
yolov5 = YOLOv5(model_path, device)

movement_model_path = "movement_network2.pt"
movement_network = MovementNetwork().cuda()
movement_network.load_state_dict(torch.load(movement_model_path), strict = False)
movement_network.eval()

print(movement_network)

prediction_results = yolov5.predict("f2098.bmp", size = 858)
from yolov5.utils.plots import image_feature
print("image_feature ", image_feature.shape)

plt.imsave('C:/Users/Chenyang/Desktop/t1/2.png', image_feature.cpu())

# image_vector = image_feature[28:118,23:193].reshape(1,-1).float()

image_vector = image_feature[4:31,5:49].reshape(1,-1).float()


print("image_vector ", image_vector.shape)
frame_window = [image_vector,image_vector,image_vector,image_vector]

image_sequence = torch.stack(frame_window,dim=1)
print("image_sequence ", image_sequence.shape)

# print("image_feature",image_feature)
# prediction_results = yolov5.predict("test2.jpg", size = 1024)
# from yolov5.utils.plots import image_feature
# print("image_feature",image_feature)
# prediction_results = yolov5.predict("test3.jpg", size = 1024)

keyboard_prediction, mouse_prediction = movement_network(image_sequence)
keyboard_prediction = torch.sigmoid(keyboard_prediction)
print("keyboard_prediction", keyboard_prediction)
print("mouse_prediction", mouse_prediction)

prediction_results.show()
