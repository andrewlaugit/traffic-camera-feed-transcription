from flask import render_template, request, flash, redirect, send_file, request, Response, session
from werkzeug.utils import secure_filename
from pathlib import Path
from app import app
import queue
from app.utils.imageextraction import *
from app.utils.car_detection import *
from app.utils.live_stream import *
from app.utils.centroidtracker import *

@app.route('/')
def home():
    
    return render_template("home.html")


##############################
#  NEEDS TO BE FIXED #########
##############################
@app.route('/uploaded_file')
def uploaded_file():
    uploaded_videos = []
    download_videos = []
    current_path = Path.cwd() / "app" / "static" / "videos"
    folder_path = current_path.iterdir()

    for file in folder_path:
        if file.is_file():
            uploaded_videos.append(file.name)

    current_path = Path.cwd() / "app" / "static" / "generatedvideos"
    folder_path = current_path.iterdir()

    for file in folder_path:
        if file.is_file():
            download_videos.append(file.name)
    
    return render_template("uploaded_file.html", video_list=uploaded_videos, video_download_list=download_videos)

@app.route('/upload', methods=["POST"])
def upload():
    file = request.files["uploaded_file"]
    file_name = secure_filename(file.filename)
    file_name = Path(file_name)
    print(file_name.suffix)

    if(file_name.suffix != ".mp4" and file_name.suffix != ".mkv"):
        flash("Wrong File Format. Upload Failed")
        return home()

    current_path = Path.cwd()
    current_path = Path(current_path)
    file_path = current_path / "app" / "static" / "videos" / file_name
    file.save(file_path.__str__())
    # image_extraction(file_path.__str__())
    return home()

@app.route('/runmodel/<file_name>')
def run_model(file_name):
    video_path = Path.cwd() / "app" / "static" / "videos" / file_name
    image_path = image_extraction_to_file(video_path.__str__())
    model = load_saved_model()
    run_model_on_file(model, image_path)
    return home()

@app.route('/youtube_video', methods=["GET", "POST"])
def youtube_video():
    if(request.method == "POST"):
        url = request.form['url']
        print(url)
        video_path = Path(get_video_youtube(url))
        image_path = image_extraction_to_file(video_path.__str__())
        model = load_saved_model()
        run_model_on_file(model, image_path)
        return home()
    else:  
        return render_template("youtube_video.html")

@app.route('/downloadfile/<file_name>')
def download_file(file_name):
    current_path = Path.cwd() / "app" / "static" / "generatedvideos"
    file_path = current_path / file_name
    return send_file(file_path.__str__(), as_attachment=True)

@app.route('/playvideo/<file_name>')
def play_video(file_name):
    # current_path = Path.cwd() / "app" / "static"
    # file_path = current_path / file_name
    print(file_name)
    # file_name = "videos" / Path(file_name)
    return render_template("play_video.html", file_name = file_name)


@app.route('/livestream', methods=["GET", "POST"])
def live_stream():
    if(request.method == "POST"):
        session["url"] = request.form['url']
        
        # get_stream(url)
        return render_template("live_stream.html")
    else:    
        return render_template("live_stream.html")

@app.route('/video_feed')
def video_feed():
    # url = "https://33-d6.divas.cloud/CHAN-286/CHAN-286_1.stream/chunklist_w1892043447.m3u8"
    # print("here")
    #Video streaming route. Put this in the src attribute of an img tag
    if(session["url"] == None):
        return None

    """
    Make single thread for segment downloads and frame extraction (to queue as pair with frame number)
    """

    """
    Make multiprocessing for running model on Frames (put into priorty queue sorted by frame number)
    The multiprocessing isn't too important as it is not time limiting on GTX 1080
    On weaker computer it may cause the video processing to run behind
    """

    """
    Change gen_frames function to only draw from priorty queue
    """

    return Response(gen_frames(session["url"]), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames( url):  # generate frame by frame from camera

    """
    Initialize first video segment and camera feed
    """
    target_fps = 10
    # last_frame = time.time()


    segment = get_segment(url)
    path, base_url = get_file_path(segment, url)
    segment_url = get_segment_url(url, base_url, segment)
    old_url = get_stream(segment_url, path, base_url, segment = segment)
    # total_frames, video_fps, frame_interval = get_frame_interval(path, segment, target_fps)
    camera = cv2.VideoCapture(path)
    model = load_saved_model()
    ct = CentroidTracker()
    frame_count = 0
    last_frame_time = 0
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            camera.release()
            segment = get_segment(url)
            path, base_url = get_file_path(segment, url)
            segment_url = get_segment_url(url, base_url, segment)

            if(segment_url != old_url):
                last_frame_time = 0
                old_url = get_stream(segment_url, path, base_url, segment = segment)
                # total_frames, video_fps, frame_interval = get_frame_interval(path, segment, target_fps)
                camera = cv2.VideoCapture(path)
                model = load_saved_model()
                frame_count = -1
        elif(camera.get(CAP_PROP_POS_MSEC) - last_frame_time >= 1000/target_fps):
            last_frame_time = camera.get(CAP_PROP_POS_MSEC)
            print(camera.get(CAP_PROP_POS_MSEC) - last_frame_time)
            frame = run_model_on_stream(model, frame, ct)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # last_frame = time.time()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        frame_count += 1
        
if __name__ == '__main__':
    app.run(debug=True)