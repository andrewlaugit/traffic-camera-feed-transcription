from cv2 import VideoCapture, imwrite, resize, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FPS, VideoWriter, VideoWriter_fourcc, imread, imencode
from pathlib import Path, PurePath
import time
import m3u8
import requests
import queue
from urllib.parse import urlparse
from app.utils.car_detection import *
from app.utils.imageextraction import video_details, check_file_path
from app.utils.centroidtracker import *


def get_segment(url):
    m3u8_segments = m3u8.load(url)
    segment = m3u8_segments.data["segments"][0]
    return segment


def get_segment_url(url, base_url, segment):
    # while(length<target_length):
    segment_url = base_url + segment["uri"]
    return segment_url


def get_file_path(segment, url):
    urltemp = url.split("/")
    base_url = ""
    for i in range(len(urltemp)-1):
        base_url = base_url + urltemp[i] + "/"

    file_name_temp = urltemp[-2].split(".")

    current_path = Path.cwd()

    save_path = current_path / "videos"

    if not Path.is_dir(save_path):
        Path.mkdir(save_path)

    current_time = str(time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime()))
    file_name = file_name_temp[0] + "_" + current_time

    save_path = save_path / file_name_temp[0]

    if not Path.is_dir(save_path):
        Path.mkdir(save_path)

    save_path = save_path.__str__() + "_" + current_time
    file_save_path = save_path.__str__() + ".mp4"

    return file_save_path, base_url


def get_stream(segment_url, file_save_path, base_url, segment, temp_url=""):

    chunk_count = 0
    request = requests.get(segment_url, stream=True)
    if(request.status_code == 200):
        # file_save_path = save_path.__str__() + "_" + str(i) + ".mp4"

        print(segment["uri"])

        with open(file_save_path, 'wb') as file:
            for chunk in request.iter_content(chunk_size=1024):
                chunk_count += 1
                # print(chunk_count)
                file.write(chunk)
                if chunk_count > 1024:
                    print(
                        'File Too Big (Greater than 1MB per .ts segment. Error Suspected. Ending.')
                    break
        file.close()
        # begin = time.time()
        # model_on_stream(model, file_save_path, duration=segment["duration"])
        # print(time.time() - begin)
        # length = length + segment["duration"]
    else:
        print("ERROR", request.status_code)
    return segment_url





def run_model_on_stream(model, image, ct, video_name="", frame_count=0, target_height=256, target_width=512, start_frame=0):
    frame = image
    image_transforms = transforms.Compose(
        [transforms.Resize((target_height, target_width)), transforms.ToTensor()])
    image = Image.fromarray(image)
    r, g, b = image.split()
    image = Image.merge("RGB", (b, g, r))
    t_image = image_transforms(image)
    img_out = draw_bounding_boxes_on_image_2(model, ct, frame)
    # img_out = cv2.cvtColor(img_out*255, cv2.COLOR_BGR2RGB)

    return img_out


def get_video_frames_details(video_path):
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

    frame_count = 0

    while(True):

        valid, _ = video.read()

        if not valid:
            # print("END at Frame Count", frame_count, " With save Interval of", save_interval)
            break

        frame_count += 1

    video.release()

    return frame_count


def get_frame_interval(video_path, segment, target_fps=10):
    total_frames = get_video_frames_details(video_path)
    video_fps = total_frames / segment["duration"]
    frame_interval = int(video_fps / target_fps)
    return total_frames, video_fps, frame_interval




def extract_frames(video_path = None, frame_queue = None, segment = None, image_per_second = 10, target_height = 256, target_width = 512, frame_count = 0):
    """
    Checks validity of provided video file path
    """
    if(check_file_path(video_path) == False):
        return False

    total_video_frames, _, _, _, _ = video_details(video_path)

    video_fps = total_video_frames / segment["duration"]

    """
    loads video
    """
    video = VideoCapture(video_path)

    """
    iterates through loaded video and extracts frames
    at intervals dictated by save_interal 
    """

    scale = (target_width, target_height)
    save_interval = int(video_fps / image_per_second)
    while(True):
        
        valid, image = video.read()

        if not valid:
            # print("END at Frame Count", frame_count, " With save Interval of", save_interval)
            break

        if(valid and frame_count % save_interval == 0):
            image = resize(image, scale)
            frame_queue.put((image, frame_count))

        frame_count += 1

    video.release()

