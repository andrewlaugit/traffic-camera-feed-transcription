from cv2 import VideoCapture, imwrite, resize, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FPS, VideoWriter, VideoWriter_fourcc, imread
import os
from os import path
from pathlib import Path
import time
from pytube import YouTube, Stream
import requests
import m3u8
from urllib.parse import urlparse


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
def image_extraction(video_path = None, image_per_second = 10, target_height = 128, target_width = 256):
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
            image_name = image_path + "\\" + video_name[0] + "_frame_{:08d}.jpg".format(frame_count)
            imwrite(image_name, image)

        frame_count += 1

    end = time.time()

    total_time = end - begin

    video.release()

    print(total_time)
    return image_path

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



def run_model_on_file(image_path):

    current_path = os.getcwd()

    if not path.isdir(current_path + r"/processed_images"):
        os.mkdir(current_path + r"/processed_images")

    video_name = image_path.split("/")

    processed_path = current_path + r"/processed_images/" + video_name[-1]

    if not path.isdir(processed_path):
        os.mkdir(processed_path)

    begin = time.time()

    image_transforms = transforms.Compose([transforms.Resize((size[0], size[1])), transforms.ToTensor()])
    for image in os.listdir(image_path):
        img = Image.open(image_path + "/" + image)
        t_image = image_transforms(img)
        img_out = draw_bounding_boxes_on_image(t_image)
        img_out = cv2.cvtColor(img_out*255,cv2.COLOR_RGB2BGR)
        out_path = processed_path + "/Processed_" + image
        imwrite(out_path, img_out)

    make_video(processed_path, size, fps)

    end = time.time()

    total_time = end - begin

    print(total_time)

def get_stream(url):
    # url = "https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1611463818/ei/KqgMYMn7B4Oqkgb5-awo/ip/69.172.154.204/id/O161GF52aKU.1/itag/301/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D299/hls_chunk_host/r2---sn-ux3n588t-t8ge.googlevideo.com/playlist_duration/30/manifest_duration/30/vprv/1/playlist_type/DVR/initcwndbps/8230/mh/1c/mm/44/mn/sn-ux3n588t-t8ge/ms/lva/mv/m/mvi/2/pl/20/dover/11/keepalive/yes/mt/1611441857/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,playlist_duration,manifest_duration,vprv,playlist_type/sig/AOq0QJ8wRAIgC0MZa5xwnAJOnwifvfnbET0GcEaCiLpY9oueKDgzm-8CIDDC2waWjsVwj6V278XlK35oedyqPg3K5fn5hMl_XjMi/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pl/lsig/AG3C_xAwRQIhAPgh1jUGt0AEWVUClGv2c2CmMnuWZ4CNlm2ojMnBP-TqAiA5CWErMunIuOIWOGJVUUl5bsqvVfkc6b4ww8o6r-RMLw%3D%3D/playlist/index.m3u8"
    # url = "https://33-d6.divas.cloud/CHAN-4859/CHAN-4859_1.stream/chunklist_w418794003.m3u8"
    # url = "https://wzmedia.dot.ca.gov/D4/E80_EOF_American_Canyon_Rd.stream/chunklist_w213026165.m3u8"
    urltemp = url.split("/")
    base_url = ""
    for i in range(len(urltemp)-1):
        base_url = base_url + urltemp[i] + "/"

    # print(base_url)

    m3u8_segments = m3u8.load(url)
    # m3u8_segments = m3u8.load("https://wzmedia.dot.ca.gov/D4/E80_EOF_American_Canyon_Rd.stream/chunklist_w213026165.m3u8")

    ts_segments = []
    length = 0

    # while(length < 60):

    for items in m3u8_segments.data['segments']:
        ts_segments.append(base_url + items['uri'])

    print(ts_segments)

    print ("Recording video...")

    print("m3u8 Duration", m3u8_segments.target_duration)

    current_time = time.localtime()

    file_name_temp = urltemp[-2].split(".")

    file_name = file_name_temp[0] + "_" + str(time.strftime("%Y_%m_%d_%H-%M-%S", current_time))

    save_path = Path.cwd() / "app" / "static" / file_name

    print(save_path)
    i = -1
    for url in ts_segments:
        print(url)
        num=0
        i = i + 1
        print("i = ", i)
        request = requests.get(url, stream=True)
        if(request.status_code == 200):
            # file_save_path = save_path.__str__() + "_" + str(i) + ".mp4" 
            file_save_path = save_path.__str__() + ".mp4" 
            with open(file_save_path,'ab') as file:
                for chunk in request.iter_content(chunk_size=1024):
                    num += 1
                    print(num)
                    file.write(chunk)
                    if num>5000:
                        print('end')
                        break
                print("here")

        else:
            print("Received unexpected status code {}".format(request.status_code))
        
        _,_,_,_, length = video_details(file_save_path)
        
    file.close()
    # image_extraction(file_save_path)

    # video_details(save_path.__str__())

    # print(total_video_frames)


# def main():
#     # video_path = r"C:\Users\Bob\Videos\Project_1.avi"

#     size = (128,256)
#     fps = 10

#     video_path = get_video_youtube("https://www.youtube.com/watch?v=MNn9qKG2UFI&ab_channel=KarolMajek")
#     image_path = image_extraction(video_path, fps, size[0], size[1])
#     gen_video_path = make_video(image_path, size, fps)

# if __name__ == '__main__':
#     main()