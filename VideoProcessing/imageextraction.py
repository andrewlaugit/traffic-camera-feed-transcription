import cv2
from cv2 import VideoCapture, imwrite, resize
import os
from os import path
import time


videopath = r"C:\Users\Bob\Videos\Project_1.avi" #####Video Source

video = VideoCapture(videopath)

total_video_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
video_fps = video.get(cv2.CAP_PROP_FPS)
print(total_video_frames)
print(frame_height)
print(frame_width)
print(video_fps)
video_length = total_video_frames / video_fps
print(video_length)

current_path = os.getcwd()

if not path.isdir(current_path + r"\images"):
    os.mkdir(current_path + r"\images")

image_path = current_path + r"\images"

frame_count = 0

image_per_second = 5 ######Number of frames extracted per second
target_height = 128
target_width = 256

scale_height = target_height/frame_height
scale_width = target_width/frame_width

scale = (target_width, target_height)

save_interval = video_fps / image_per_second

begin = time.time()

while(frame_count < total_video_frames):
    
    valid, image = video.read()

    image = resize(image, scale)

    if not valid:
        print("END")
        break

    if(valid and frame_count % save_interval == 0):
        image_name = image_path + r"\test_frame" + str(frame_count) +r".jpg"
        imwrite(image_name, image)

    frame_count += 1

end = time.time()

total_time = end - begin

print(total_time)

