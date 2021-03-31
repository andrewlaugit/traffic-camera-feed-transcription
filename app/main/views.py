from flask import render_template, request, flash, redirect, send_file, request, Response, session
from werkzeug.utils import secure_filename
from pathlib import Path
from app import app
import queue
import threading
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

    current_path = Path.cwd() / "app" / "static" #/ "generatedvideos"
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

    if(file_name.suffix != ".mp4" and file_name.suffix != ".mkv" and file_name.suffix != ".avi"):
        flash("Wrong File Format. Upload Failed")
        return home()

    current_path = Path.cwd()
    current_path = Path(current_path)
    file_path = current_path / "app" / "static" / "videos" / file_name
    file.save(file_path.__str__())
    return home()

@app.route('/runmodel/<file_name>')
def run_model(file_name):
    frame_queue = queue.Queue()
    processed_queue = queue.Queue()
    video_path = Path.cwd() / "app" / "static" / "videos" / file_name
    image_extraction_to_queue(video_path.__str__(), frame_queue, image_per_second=20, save_images = "on")
    model = load_saved_model()
    ct = CentroidTracker()
    run_model_on_queue(model, ct, frame_queue, processed_queue, video_path=video_path, fps = 20, save_images="on")
    make_video_from_queue(file_name, processed_queue, (512, 256), 20)
    return uploaded_file()

@app.route('/youtube_video', methods=["GET", "POST"])
def youtube_video():
    if(request.method == "POST"):
        url = request.form['url']
        # print(url)
        video_path = Path(get_video_youtube(url))
        file_name = video_path.stem
        frame_queue = queue.Queue()
        processed_queue = queue.Queue()
        image_extraction_to_queue(video_path.__str__(), frame_queue)
        model = load_saved_model()
        ct = CentroidTracker()
        run_model_on_queue(model, ct, frame_queue, processed_queue)
        make_video_from_queue(file_name.__str__(), processed_queue, (512, 256), 10)
        return home()
    else:  
        return render_template("youtube_video.html")

@app.route('/downloadfile/<file_name>')
def download_file(file_name):
    current_path = Path.cwd() / "app" / "static"
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

    target_fps = 20

    """
    Make single thread for segment downloads and frame extraction (to queue as pair with frame number)
    """
    frame_queue = queue.Queue()

    sf = threading.Thread(target=get_stream_and_frames, args=(session["url"], frame_queue, target_fps,), daemon=True)
    sf.start()

    """
    Using Threads instead of Multiprocessing for now
    Make multiprocessing for running model on Frames (put into priorty queue sorted by frame number)
    The multiprocessing isn't too important as it is not time limiting on GTX 1080
    On weaker computer it may cause the video processing to run behind
    """
    processed_queue = queue.PriorityQueue()
    rm = threading.Thread(target=run_model_on_queue_loop, args=(frame_queue, processed_queue, target_fps,), daemon=True)
    rm.start()

    return Response(gen_frames(processed_queue, target_fps), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames(processed_queue, target_fps):  # generate frame by frame from camera

    """
    Initialize first video segment and camera feed
    """
    last_frame_shown_time = time.time()
    while True:
        count, frame = processed_queue.get()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        while(time.time() - last_frame_shown_time < (1 / target_fps)):
            time.sleep((1/(target_fps*4)))

        last_frame_shown_time = time.time()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


        
if __name__ == '__main__':
    app.run(debug=True)