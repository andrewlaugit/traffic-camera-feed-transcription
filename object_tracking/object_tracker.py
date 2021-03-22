## python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

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
# cap = cv2.VideoCapture('1atStateParkDr_2021_02_01_13-02-34.mp4')
# cap = cv2.VideoCapture('1atEofCapitolaAve_2021_02_01_13-49-10.mp4')

height = 256
width = 512

cap = cv2.VideoCapture('highwaycar.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('video.avi', fourcc, fps,(width,height))

car_past_direction = dict()
count = 0

while cap.isOpened():
	ret, frame = cap.read()
	print("time stamp current frame:", count / fps)

	frame = cv2.resize(frame, (width, height))
	rects = []
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	image_transforms = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
	t_image = image_transforms(Image.fromarray(img))

	img_out, contours = draw_bounding_boxes_on_image(t_image, model)
	for cnt in contours:
		# draw a bounding box surrounding the object so we can
		# visualize it
		(startX, startY, endX, endY) = cv2.boundingRect(cnt)
		if cv2.contourArea(cnt) > 0.6 * endX * endY and  cv2.contourArea(cnt) > 150:
		# if cv2.contourArea(cnt) > 150:
			endX = startX + endX
			endY = startY + endY
			if (endX >= 0 and endX <= width and endY >=0 and endY <= height and startX >=0 and startX <= width and startY >=0 and startY <= height):
				rects.append((startX, startY, endX, endY))

				cv2.rectangle(frame, (startX, startY),(endX, endY),
							  (0, 255, 0), 1)

	objects, object_direction,road_directions = ct.update(rects, count/fps)
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "Car {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
		cv2.circle(frame, (centroid[0], centroid[1]), 1, (0, 255, 0), 1)

		if object_direction[objectID] != 0:
			cv2.arrowedLine(frame, (centroid[0], centroid[1]), (centroid[0]+object_direction[objectID][0], centroid[1]+object_direction[objectID][1]),
							(255,0,0), 1, tipLength=0.5)
			if objectID not in car_past_direction:
				car_past_direction[objectID] = []



		# img_out = np.transpose(img_out, (2, 0, 1))
	# plt.imshow(img_out)
	# plt.show()
	frame = cv2.resize(frame, (width, height))

	# total_num_of_car = ct.numberOfObjects()
	# totalCarText = "Total # of Cars: "+ str(total_num_of_car)
	# currentCarText = "Current # of Cars: " + str(len(objects))
	# cv2.putText(frame, totalCarText, (0, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
	# cv2.putText(frame, currentCarText, (0, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

	trafficColor = (0, 255, 0) # green

	if (len(objects) > 10): # Red
		trafficColor = (0, 0, 255)
	elif (len(objects) > 7): # Orange
		trafficColor = (0, 144, 255)

	# Traffic Conditions
	# cv2.putText(frame, "Traffic Condition", (400, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
	# cv2.rectangle(frame, (400, 230), (512, 255), trafficColor, -1)

	cv2.rectangle(frame, (390, 180), (512, 256), (255, 255, 255), -1)

	if (len(road_directions) >0):
		cv2.putText(frame, "Direction", (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

		cv2.arrowedLine(frame, (480, 190),
						(480+ int(road_directions[0][0][0]*15),190 +  int(road_directions[0][0][1]*15)),
						(0, 0, 0), 1, tipLength=0.5)

		cv2.putText(frame, str(int(road_directions[0][1])), (495, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

		if (len(road_directions) > 1):
			cv2.putText(frame, "Direction", (400, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

			cv2.arrowedLine(frame, (480, 220),
							(480 + int(road_directions[1][0][0] * 15), 220 + int(road_directions[1][0][1] * 15)),
							(0, 0, 0), 1, tipLength=0.5)
			cv2.putText(frame, str(road_directions[1][1]), (495, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

		# if (len(road_directions) > 2):
		# 	cv2.putText(frame, "Direction", (400, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
		#
		# 	cv2.arrowedLine(frame, (480, 240),
		# 					(480 + int(road_directions[2][0][0] * 15), 240 + int(road_directions[2][0][1] * 15)),
		# 					(255, 0, 0), 1, tipLength=0.5)
		# 	cv2.putText(frame, str(road_directions[2][1]), (495, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)		#cv2.rectangle(frame, (400, 230), (512, 255), trafficColor, -1)


	cv2.imshow("frame", frame)
	video.write(frame)

	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	count += 1

cv2.destroyAllWindows()
video.release()
