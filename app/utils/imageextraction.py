from cv2 import VideoCapture, CAP_PROP_POS_MSEC, imwrite, resize, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FPS, VideoWriter, VideoWriter_fourcc, imread, imencode
import os
from os import path
from pathlib import Path
import time
from numpy.lib.type_check import imag
from pytube import YouTube, Stream
import requests
import m3u8
import queue
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
Extracts frames from the video at a selected frame rate saves images in folder
"""


def image_extraction_to_file(video_path, image_per_second=10, target_height=256, target_width=512):
    """
    Checks validity of provided video file path
    """
    if(check_file_path(video_path) == False):
        return False

    """
    loads video
    """
    video = VideoCapture(video_path)

    if not video.isOpened():
        print("Unable to open file. Please ensure input video is either .AVI or .MKV")
        return False

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

    begin = time.time()

    """
    iterates through loaded video and extracts frames
    at intervals dictated by save_interal 
    """

    scale = (target_width, target_height)
    last_frame_time = -10000000
    frame_count = 0
    while True:

        valid, image = video.read()

        if not valid:
            break

        if(video.get(CAP_PROP_POS_MSEC) - last_frame_time >= 1000 / image_per_second):
            last_frame_time = video.get(CAP_PROP_POS_MSEC)
            image = resize(image, scale)
            image_name = image_path / (video_name.__str__() + "_frame_{:08d}.jpg".format(frame_count))
            imwrite(image_name.__str__(), image)

        frame_count += 1

    end = time.time()

    total_time = end - begin

    video.release()

    print("Total time to extract and save frames: ", total_time)
    return image_path


"""
Extracts frames from the video at a selected frame rate saves images to queue
"""
def image_extraction_to_queue(video_path, frame_queue, image_per_second=10, target_height=256, target_width=512, frame_count = 0):
    """
    Checks validity of provided video file path
    """
    if(check_file_path(video_path) == False):
        return False

    """
    loads video
    """
    video = VideoCapture(video_path)

    if not video.isOpened():
        print("Unable to open file. Please ensure input video is either .AVI or .MKV")
        return False

    """
    iterates through loaded video and extracts frames
    at intervals dictated by save_interal 
    """

    scale = (target_width, target_height)
    last_frame_time = -10000000
    while True:

        valid, image = video.read()

        if not valid:
            # print("END at Frame Count", frame_count, " With save Interval of", save_interval)
            break

        if(video.get(CAP_PROP_POS_MSEC) - last_frame_time >= 1000 / image_per_second):
            last_frame_time = video.get(CAP_PROP_POS_MSEC)
            image = resize(image, scale)
            frame_queue.put((frame_count, image))

        frame_count += 1

    video.release()

    return frame_count





def run_model_on_file(model, image_path, target_height=256, target_width=512, start_frame=0):

    current_path = Path.cwd()

    processed_path = current_path / "processed_images"

    if not Path.is_dir(processed_path):
        Path.mkdir(processed_path)

    video_name = image_path.stem

    processed_path = processed_path / video_name

    if not Path.is_dir(processed_path):
        Path.mkdir(processed_path)

    begin = time.time()

    ct = CentroidTracker()
    for image_name in Path.iterdir(image_path):
        image_name = image_name.name
        img = imread((image_path / image_name).__str__())
        img_out = draw_bounding_boxes_on_image_2(model, ct, img)
        out_path = processed_path / ("Processed_" + image_name.__str__())
        imwrite(out_path.__str__(), img_out)

    end = time.time()

    total_time = end - begin

    print("Time taken to process video frames:", total_time)

    return processed_path


def run_model_on_queue(model, frame_queue, processed_queue, target_height=256, target_width=512, start_frame=0):
    begin = time.time()
    ct = CentroidTracker()
    while frame_queue.empty() is False:
        frame_num, img = frame_queue.get()
        img_out = draw_bounding_boxes_on_image_2(model, ct, img)
        processed_queue.put((frame_num, img_out))

    end = time.time()

    total_time = end - begin

    print("Time taken to process video frames:", total_time)


"""
Generates video from images in given file
"""


def make_video(image_path, size = (512, 256), fps = 10):

    current_path = Path.cwd()

    save_path = current_path / "app" / "static" / "generatedvideos"

    images = []
    names = []

    for image in Path.iterdir(image_path):
        names.append(image.stem)
        img = imread(image.__str__())
        images.append(img)

    out = [x for _, x in sorted(zip(names, images))]

    if not Path.is_dir(save_path):
        Path.mkdir(save_path)

    video_name = image_path.stem + ".avi"

    save_path = save_path / video_name

    codex = VideoWriter_fourcc(*'XVID')
    writer = VideoWriter(save_path.__str__(), codex, fps, (size[0], size[1]))
    for image in out:
        writer.write(image)
    writer.release()

    return save_path


def make_video_from_queue(video_name, processed_queue, size = (512, 256), fps = 10):
    
    current_path = Path.cwd()

    images = []
    names = []

    while processed_queue.empty() is False:
        name, image = processed_queue.get()
        names.append(name)
        images.append(image)

    out = [x for _, x in sorted(zip(names, images))]

    save_path = current_path / "app" / "static" / "generatedvideos"

    if not Path.is_dir(save_path):
        Path.mkdir(save_path)

    video_name = video_name + ".avi"

    save_path = save_path / video_name

    codex = VideoWriter_fourcc(*'XVID')
    writer = VideoWriter(save_path.__str__(), codex, fps, (size[0], size[1]))

    for image in out:
        # cv2_imshow(image)
        writer.write(image)
    writer.release()

    return save_path
