from cv2 import VideoCapture, imwrite, resize, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FPS, VideoWriter, VideoWriter_fourcc, imread, imencode
from pathlib import Path, PurePath
import time
import m3u8
import requests
import queue
from moviepy.editor import VideoFileClip
from urllib.parse import urlparse
from app.utils.car_detection import *
from app.utils.imageextraction import image_extraction_to_queue, run_model_on_queue
from app.utils.centroidtracker import *


def get_segment(url):
    m3u8_segments = m3u8.load(url)
    segment = m3u8_segments.data["segments"][0]
    return segment

def get_segments(url):
    m3u8_segments = m3u8.load(url)
    segments = m3u8_segments.data["segments"]
    return segments

def get_segment_url(base_url, segment):
    segment_url = base_url + segment["uri"]
    return segment_url


def get_file_path(url):
    urltemp = url.split("/")

    file_name_temp = urltemp[-2].split(".")

    current_path = Path.cwd()

    save_path = current_path / "videos"

    if not Path.is_dir(save_path):
        Path.mkdir(save_path)

    current_time = str(time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime()))

    save_path = save_path / file_name_temp[0]

    if not Path.is_dir(save_path):
        Path.mkdir(save_path)

    save_path = save_path.__str__() + "_" + current_time + ".mp4"

    return save_path


def get_base_url(url):
    urltemp = url.split("/")
    base_url = ""
    for i in range(len(urltemp)-1):
        base_url = base_url + urltemp[i] + "/"
    return base_url


def get_stream(segment_url, file_save_path):
    request = requests.get(segment_url, stream=True)
    if(request.status_code == 200):
        with open(file_save_path, 'wb') as file:
            for chunk in request.iter_content(chunk_size=1024):
                file.write(chunk)
        file.close()
    else:
        print("ERROR", request.status_code)
    return segment_url

def get_stream_and_frames(url, frame_queue, target_fps = 10, run_time = 300):
    segments = queue.Queue()
    segment_urls = queue.Queue()
    all_segments = []

    video_time = 0

    base_url = get_base_url(url)

    size = (512, 256)

    total_frames = 0

    while True:
        #forces function to exit after set time
        if video_time > run_time:
            exit()

        if len(all_segments) > 10:
            all_segments.pop(0)

        new_segments = get_segments(url)

        for item in new_segments:
            if item not in all_segments:
                segments.put(item)
                all_segments.append(item)

        while segments.empty() is False:
            segment_urls.put(get_segment_url(base_url, segments.get()))

        if segment_urls.empty() is False:
            save_path = get_file_path(url)
            get_stream(segment_urls.get(), save_path)
            video_time = video_time + VideoFileClip(save_path).duration
            total_frames = image_extraction_to_queue(save_path, frame_queue, target_height = size[1], target_width= size[0],frame_count = total_frames, image_per_second = target_fps)
        else:
            time.sleep(1)
    
def run_model_on_queue_loop(frame_queue, processed_queue, fps, video_path="", run_time = 300,  save_images="off"):
    start_time = time.time()
    model = load_saved_model()
    if video_path != "":
        video_name = video_path.stem
        ct = CentroidTracker(save_file_name=video_name.__str__())
    else:
        ct = CentroidTracker()
    while True:
        #forces function to exit after set time
        if time.time() - start_time > run_time and frame_queue.empty() is True:
            exit()

        if frame_queue.empty() is False:
            run_model_on_queue(model, ct, frame_queue, processed_queue, fps = fps, video_path=video_path, save_images=save_images)
        else:
            time.sleep(1)

    