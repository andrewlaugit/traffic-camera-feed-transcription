from cv2 import VideoCapture, imwrite, resize, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FPS, VideoWriter, VideoWriter_fourcc, imread
import os
from os import path
from pathlib import Path
import time
from pytube import YouTube, Stream
import requests
import m3u8
from urllib.parse import urlparse
from app.utils.car_detection import *


"""
Retrieves video from youtube in smallest mp4 format available and saves it to videos folder in working directory
"""
def get_video_youtube(url):
    """
    ensures video folder exists in project root directory
    """

    save_path = Path.cwd() / "app" / "static" 
    # save_path = current_path + r"\videos"
    
    if not path.isdir(save_path):
        os.mkdir(save_path)

    """
    Gets youtube object, identifies the smallest stream and downloads it
    smallest stream is selected as our model is being trained on low resolution images and this saves space
    """
    youtube_object = YouTube(url)
    video_path = youtube_object.streams.filter(progressive=True, file_extension='mp4').get_lowest_resolution().download(save_path.__str__())

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
Extracts frames from the video at a selected frame rate
"""
def image_extraction(video_path = None, image_per_second = 10, target_height = 128, target_width = 256, frame_count = 0):
    """
    Checks validity of provided video file path
    """
    if(check_file_path(video_path) == False):
        return False

    total_video_frames, video_fps, _, _, _ = video_details(video_path)

    """
    loads video
    """
    video = VideoCapture(video_path)

    if not video.isOpened():
        print("Unable to open file. Please ensure input video is either .AVI or .MKV")
        return False

    _, video_name = os.path.split(video_path)

    video_name = video_name.split(".")

    """
    gets working directory
    """
    current_path = os.getcwd()

    """
    ensures image folder exists in project root directory
    """
    if not path.isdir(current_path + r"\images"):
        os.mkdir(current_path + r"\images")

    image_path = current_path + r"\images\\" + video_name[0]

    if not path.isdir(image_path):
        os.mkdir(image_path)    

    print(image_path)


    begin = time.time()

    """
    iterates through loaded video and extracts frames
    at intervals dictated by save_interal 
    """
    
    scale = (target_width, target_height)
    # save_interval = int(video_fps / image_per_second)
    while(frame_count < total_video_frames):
        
        valid, image = video.read()

        if not valid:
            # print("END at Frame Count", frame_count, " With save Interval of", save_interval)
            break

        # if(valid and frame_count % save_interval == 0):
        if(valid):
            image = resize(image, scale)
            image_name = image_path + "\\" + video_name[0] + "_frame_{:08d}.jpg".format(frame_count)
            imwrite(image_name, image)

        frame_count += 1

    end = time.time()

    total_time = end - begin

    video.release()

    print(total_time)
    return image_path, frame_count

"""
Generates video from images in given file
"""
def make_video(image_path, size, fps):
    
    current_path = os.getcwd()

    images = []
    names = []

    for image in os.listdir(image_path):
        names.append(image)
        img = imread(os.path.join(image_path, image))
        images.append(img)

    out = [x for _, x in sorted(zip(names,images))]

    if not path.isdir(current_path + r"\generatedvideos"):
        os.mkdir(current_path + r"\generatedvideos")

    name = image_path.split("\\")
    print(name[len(name)-1])

    video_path = current_path + r"\generatedvideos\\" + name[len(name)-1] + r".avi"

    codex = VideoWriter_fourcc(*'MP4V')
    writer = VideoWriter(video_path, codex, fps, (size[1],size[0]))

    for image in out:
        # cv2_imshow(image)
        writer.write(image)
    writer.release()

    return video_path



def run_model_on_file(model, image_path, target_height = 128, target_width = 256, start_frame = 0):

    current_path = os.getcwd()

    if not path.isdir(current_path + r"/processed_images"):
        os.mkdir(current_path + r"/processed_images")

    video_name = image_path.split("\\")

    processed_path = current_path + r"/processed_images/" + video_name[-1]

    if not path.isdir(processed_path):
        os.mkdir(processed_path)

    begin = time.time()

    image_transforms = transforms.Compose([transforms.Resize((target_height, target_width)), transforms.ToTensor()])
    for image in os.listdir(image_path):
        img = Image.open(image_path + "/" + image)
        t_image = image_transforms(img)
        img_out = draw_bounding_boxes_on_image(model, t_image)
        img_out = cv2.cvtColor(img_out*255,cv2.COLOR_RGB2BGR)
        out_path = processed_path + "/Processed_" + image
        imwrite(out_path, img_out)

    make_video(processed_path, size, fps)

    end = time.time()

    total_time = end - begin

    print(total_time)


def get_segment(url):
    m3u8_segments = m3u8.load(url)
    segment = m3u8_segments.data["segments"][0]['uri']
    return segment

def get_stream(url, target_length = 20):
    urltemp = url.split("/")
    base_url = ""
    for i in range(len(urltemp)-1):
        base_url = base_url + urltemp[i] + "/"

    length = 0

    current_time = time.localtime()

    file_name_temp = urltemp[-2].split(".")



    temp_url = ""

    frame_count = 0

    while(length<target_length):
        segment_url = get_segment(url)
        segment_url = base_url + segment_url

        if(segment_url != temp_url):
            chunk_count=0
            request = requests.get(segment_url, stream=True)
            if(request.status_code == 200):
                # file_save_path = save_path.__str__() + "_" + str(i) + ".mp4" 
                file_name = file_name_temp[0] # + "_" + str(time.strftime("%Y_%m_%d_%H-%M-%S", current_time))

                save_path = Path.cwd() / "app" / "static" / file_name
                file_save_path = save_path.__str__() + ".mp4" 
                print(file_save_path)
                with open(file_save_path,'wb') as file:
                    for chunk in request.iter_content(chunk_size=1024):
                        chunk_count += 1
                        # print(chunk_count)
                        file.write(chunk)
                        if chunk_count>1000:
                            print('File Too Big (Greater than 1MB per .ts segment. Error Suspected. Ending.')
                            break
                    print("here")
                file.close()
                _, frame_count = image_extraction(file_save_path, frame_count=frame_count)
                print("Frame Count =", frame_count)
                _,_,_,_, seg_length = video_details(file_save_path)
                length = length + seg_length
            else:
                print("ERROR", request.status_code)
        temp_url = segment_url
        
        
        print("Frame Count =", frame_count)
        
    
    # image_extraction(file_save_path)

    # print(total_video_frames)