from cv2 import VideoCapture, CAP_PROP_POS_MSEC, imwrite, resize, cvtColor, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FPS, VideoWriter, VideoWriter_fourcc, imread, imencode
import os
from os import path
from pathlib import Path
import time
from numpy.lib.type_check import imag
from pytube import YouTube, Stream
import requests
import m3u8
import queue
from moviepy.editor import VideoFileClip
from urllib.parse import urlparse
from app.utils.car_detection import *


"""
Retrieves video from youtube in smallest mp4 format available and saves it to videos folder in working directory
"""


def get_video_youtube(url, save_path=Path.cwd() / "app" / "static" / "videos"):
    """
    ensures video folder exists in project root directory
    """

    if not Path.is_dir(save_path):
        Path.mkdir(save_path)

    """
    Gets youtube object, identifies the smallest stream and downloads it
    smallest stream is selected as our model is being trained on low resolution images and this saves space
    """
    youtube_object = YouTube(url)
    video_path = youtube_object.streams.filter(
        progressive=True, file_extension='mp4').get_lowest_resolution().download(save_path.__str__())

    return video_path


"""
Checks that the path goes to a file
"""


def check_file_path(video_path):
    if video_path == None:
        print("No video file provided")
        return False

    if not path.exists(video_path):
        print("Invalid Path")
        return False

    if not path.isfile(video_path):
        print("Input must be file")
        return False

    return True


"""
Get and reutrns the details for the video pointed to by the file path
"""


def video_details(video_path):
    check_file_path(video_path)
    # print(type(video_path))
    video = VideoCapture(video_path)

    if not video.isOpened():
        print("Unable to open file. Please ensure input video is either .AVI or .MKV")
        return False

    total_video_frames = video.get(CAP_PROP_FRAME_COUNT)
    frame_height = video.get(CAP_PROP_FRAME_HEIGHT)
    frame_width = video.get(CAP_PROP_FRAME_WIDTH)
    video_fps = video.get(CAP_PROP_FPS)
    video_length = total_video_frames / video_fps
    print("Total Number of Frames: ", total_video_frames)
    print("Frame Height: ", frame_height)
    print("Frame Width: ", frame_width)
    print("Frames per Second: ", video_fps)
    print("Video Length: ", video_length)

    video.release()

    return total_video_frames, video_fps, frame_height, frame_width, video_length


"""
Extracts frames from the video at a selected frame rate saves images to queue
"""


def image_extraction_to_queue(video_path, frame_queue, image_per_second=10, target_height=256, target_width=512, frame_count=0, save_images = "off"):
    """
    Checks validity of provided video file path
    """

    start_time = time.time()

    if(check_file_path(video_path) == False):
        return False

    """
    loads video
    """
    clip = VideoFileClip(video_path)
    video_length = clip.duration
    print("Video Length According to MOVIEPY: ", video_length)

    """
    iterates through loaded video and extracts frames
    at intervals dictated by save_interal 
    """
    if save_images == "on":
        video_path = Path(video_path)

        video_name = video_path.stem

        """
        gets working directory
        """
        current_path = Path.cwd()

        """
        ensures image folder exists in project root directory
        """

        image_path = current_path / "images"

        if not Path.is_dir(image_path):
            Path.mkdir(image_path)

        image_path = image_path / video_name

        if not Path.is_dir(image_path):
            Path.mkdir(image_path)

    scale = (target_width, target_height)
    image_transforms = transforms.Compose(
        [transforms.Resize((target_height, target_width)), transforms.ToTensor()])

    time_inc = 1 / image_per_second

    for vid_time in np.arange(0, video_length, time_inc):

        frame = clip.get_frame(vid_time)
        frame = cv2.resize(frame, scale)
        if frame.ndim != 3:
            frame = np.expand_dims(frame, axis = 2)
            frame = np.repeat(frame, 3, axis = 2)
            print("Grayscale changed back to", frame.shape)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t_image = image_transforms(Image.fromarray(img))
        frame_queue.put((frame_count, img, t_image))
        if save_images == "on":
            image_name = image_path / (video_name.__str__() + "_frame_{:08d}.jpg".format(frame_count))
            imwrite(image_name.__str__(), img)
        frame_count += 1

    print("Image Extraction Took:", time.time() - start_time, " seconds")

    return frame_count


def run_model_on_queue(model, ct, frame_queue, processed_queue, fps = 20, video_path = "", save_images = "off"):
    begin = time.time()

    video_name = Path(video_path).stem

    save_path = Path.cwd() / "processed_images"

    if not Path.is_dir(save_path):
        Path.mkdir(save_path)

    save_path = Path.cwd() / "processed_images" / video_name

    if not Path.is_dir(save_path):
        Path.mkdir(save_path)

    while frame_queue.empty() is False:
        frame_num, frame, t_image = frame_queue.get()
        img_out = count_vehicles(model, ct, frame, t_image, frame_num, fps)
        processed_queue.put((frame_num, img_out))
        if save_images == "on":
            image_name = save_path / (video_name.__str__() + "_frame_{:08d}.jpg".format(frame_num))
            imwrite(image_name.__str__(), img_out)

    end = time.time()

    total_time = end - begin

    print("Time taken to process video frames:", total_time)


"""
Generates video from images in given file
"""

def make_video_from_queue(video_name, processed_queue, size=(512, 256), fps=10):

    current_path = Path.cwd()

    save_path = current_path / "app" / "static" 

    if not Path.is_dir(save_path):
        Path.mkdir(save_path)

    video_name = Path(video_name).stem + ".mp4"

    save_path = save_path / video_name

    codex = VideoWriter_fourcc(*'mp4v')
    writer = VideoWriter(save_path.__str__(), codex, fps, (size[0], size[1]))

    while processed_queue.empty() is False:
        _, image = processed_queue.get()
        writer.write(image)
    writer.release()

    return save_path
