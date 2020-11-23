import cv2
from cv2 import VideoCapture, imwrite, resize
import os
from os import path
import time



# requires that video be AVI format
def image_extraction(video_path = None, image_per_second = 5, target_height = 128, target_width = 256):

    if video_path == None:
        print("No video file provided")
        return False

    if not path.exists(video_path):
        print("Invalid Path")
        return False

    if not path.isfile(video_path):
        print("Input must be file")
        return False

    _, video_name = os.path.split(video_path)

    video_name = video_name.split(".")

    if(video_name[1] != "avi" or video_name[1] != "mkv"):
        print("Invalid File Format. Inputs most be AVI")
        return False

    video = VideoCapture(video_path)

    if not video.isOpened():
        print("Unable to open file. Please ensure input video is either .AVI or .MKV")
        return False

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

    scale = (target_width, target_height)

    save_interval = video_fps / image_per_second

    begin = time.time()

    while(frame_count < total_video_frames):
        
        valid, image = video.read()

        if not valid:
            print("END")
            break

        if(valid and frame_count % save_interval == 0):
            image = resize(image, scale)
            image_name = image_path + "\\" + video_name[0] + "_frame_" + str(frame_count) +r".jpg"
            imwrite(image_name, image)

        frame_count += 1

    end = time.time()

    total_time = end - begin

    video.release()

    print(total_time)


def main():
    video_path = r"C:\Users\Bob\Videos\Introduction to ML - 1 - What is a ML task.mp4"

    image_extraction(video_path)

if __name__ == '__main__':
    main()