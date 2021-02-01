# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from tracker.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import torch
import cv2
from cv2 import VideoCapture, imwrite, resize, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FPS, VideoWriter, VideoWriter_fourcc, imread
from model import draw_bounding_boxes_on_image, TrafficCamera_Vgg
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import os
from os import path
import matplotlib.pyplot as plt
# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()

# intialize the model
model = TrafficCamera_Vgg()
model.load_state_dict(torch.load("vgg_augmented_16_30_0.001_best",map_location=torch.device('cpu')))
model.eval()

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")

"""
Checks that the path goes to a file
"""

# cap = cv2.VideoCapture('4K Road traffic video for object detection and tracking - free download now!.avi')
cap = cv2.VideoCapture('Relaxing highway traffic.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('video.avi', fourcc, 15,(512,256))

while cap.isOpened():
	ret, frame = cap.read()
	frame = cv2.resize(frame, (512, 256))
	rects = []
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	image_transforms = transforms.Compose([transforms.Resize((256, 512)), transforms.ToTensor()])
	t_image = image_transforms(Image.fromarray(img))
	# img = np.transpose(img, (2, 0, 1))
	# tensor_img = torch.from_numpy(img)
	# tensor_img = tensor_img.float()
	img_out, contours = draw_bounding_boxes_on_image(t_image, model)
	for cnt in contours:
		# draw a bounding box surrounding the object so we can
		# visualize it
		(startX, startY, endX, endY) = cv2.boundingRect(cnt)
		if cv2.contourArea(cnt) > 0.6 * endX * endY \
				or (endX * 3 > endY and endY * 3 > endX and endX * endY > 200):
			endX = startX + endX
			endY = startY + endY
			rects.append((startX, startY, endX, endY))

			cv2.rectangle(frame, (startX, startY),(endX, endY),
						  (0, 255, 0), 1)

	objects = ct.update(rects)
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "Car {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
		cv2.circle(frame, (centroid[0], centroid[1]), 1, (0, 255, 0), 1)

	# img_out = np.transpose(img_out, (2, 0, 1))
	# plt.imshow(img_out)
	# plt.show()
	frame = cv2.resize(frame, (512, 256))

	total_num_of_car = ct.numberOfObjects()
	totalCarText = "Total # of Cars: "+ str(total_num_of_car)
	currentCarText = "Current # of Cars: " + str(len(objects))
	cv2.putText(frame, totalCarText, (0, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
	cv2.putText(frame, currentCarText, (0, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

	trafficColor = (0, 255, 0) # green

	if (len(objects) > 10): # Red
		trafficColor = (0, 0, 255)
	elif (len(objects) > 7): # Orange
		trafficColor = (0, 144, 255)

	cv2.putText(frame, "Traffic Condition", (400, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
	cv2.rectangle(frame, (400, 230), (512, 255), trafficColor, -1)

	cv2.imshow("frame", frame)
	video.write(frame)

	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


cv2.destroyAllWindows()
video.release()
