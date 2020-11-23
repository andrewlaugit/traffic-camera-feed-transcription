from cv2 import VideoCapture, imwrite, resize, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FPS
import os
from os import path
import time
from pytube import YouTube, Stream

"""
Retrieves video from youtube in smallest mp4 format available and saves it to videos folder in working directory
"""
def get_video_youtube(url):

    current_path = os.getcwd()

    """
    ensures video folder exists in project root directory
    """
    if not path.isdir(current_path + r"\videos"):
        os.mkdir(current_path + r"\videos")

    save_path = current_path + r"\videos"

    """
    Gets youtube object, identifies the smallest stream and downloads it
    """
    youtube_object = YouTube(url)
    video_path = youtube_object.streams.filter(progressive=True, file_extension='mp4').get_lowest_resolution().download(save_path)

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

    video = VideoCapture(video_path)

    if not video.isOpened():
        print("Unable to open file. Please ensure input video is either .AVI or .MKV")
        return False

    total_video_frames = video.get(CAP_PROP_FRAME_COUNT)
    frame_height = video.get(CAP_PROP_FRAME_HEIGHT)
    frame_width = video.get(CAP_PROP_FRAME_WIDTH)
    video_fps = video.get(CAP_PROP_FPS)
    video_length = total_video_frames / video_fps
    print(total_video_frames)
    print(frame_height)
    print(frame_width)
    print(video_fps)  
    print(video_length)

    video.release()

    return total_video_frames, video_fps, frame_height, frame_width, video_length


"""
Extracts frames from the video at a selected frame rate
"""
def image_extraction(video_path = None, image_per_second = 5, target_height = 128, target_width = 256):
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

    image_path = current_path + r"\images"

    begin = time.time()

    """
    iterates through loaded video and extracts frames
    at intervals dictated by save_interal 
    """
    frame_count = 0
    scale = (target_width, target_height)
    save_interval = int(video_fps / image_per_second)
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
    # video_path = r"C:\Users\Bob\Videos\Project_1.avi"

    video_path = get_video_youtube("https://www.youtube.com/watch?v=MNn9qKG2UFI&ab_channel=KarolMajek")
    image_extraction(video_path)

if __name__ == '__main__':
    main()