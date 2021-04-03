from flask import render_template, request, flash, redirect, send_file, request, Response, session, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
from app import app
import queue
# import json
import threading
from app.utils.imageextraction import *
from app.utils.car_detection import *
from app.utils.live_stream import *
from app.utils.centroidtracker import *

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


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
    session["file_name"] = file_name
    session["traffic_summary"] = []
    return render_template("model_stream.html", traffic_summary = session["traffic_summary"])

@app.route('/run_model_uploaded_stream')
def run_model_stream():
    file_name = session['file_name']
    frame_queue = queue.Queue()
    processed_queue = queue.Queue()
    video_path = Path.cwd() / "app" / "static" / "videos" / file_name

    target_fps = 20
    min_run_time = 60

    """
    Make single thread for segment downloads and frame extraction (to queue as pair with frame number)
    """
    sf = threading.Thread(target=image_extraction_to_queue, args=(video_path.__str__(), frame_queue, target_fps, 256, 512, 0, "off",), daemon=True)
    sf.start()

    """
    Using Threads instead of Multiprocessing for now
    Make multiprocessing for running model on Frames (put into priorty queue sorted by frame number)
    The multiprocessing isn't too important as it is not time limiting on GTX 1080
    On weaker computer it may cause the video processing to run behind
    """
    rm = threading.Thread(target=run_model_on_queue_loop, args=(frame_queue, processed_queue, target_fps, video_path, min_run_time, "off",), daemon=True)
    rm.start()

    return Response(gen_frames(processed_queue, target_fps, min_run_time), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/youtube_video', methods=["GET", "POST"])
def youtube_video():
    if(request.method == "POST"):
        url = request.form['url']
        video_path = Path(get_video_youtube(url))
        file_name = video_path.stem
        return run_model(file_name)
    else:  
        return render_template("youtube_video.html")

@app.route('/downloadfile/<file_name>')
def download_file(file_name):
    current_path = Path.cwd() / "app" / "static"
    file_path = current_path / file_name
    return send_file(file_path.__str__(), as_attachment=True)


@app.route('/livestream', methods=["GET", "POST"])
def live_stream():
    if(request.method == "POST"):
        session["url"] = request.form['url']
        
        # get_stream(url)
        return render_template("live_stream.html")
    else:    
        return render_template("live_stream.html")

@app.route('/live_video_feed')
def live_video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    if(session["url"] == None):
        return None

    target_fps = 20
    run_time = 60

    """
    Make single thread for segment downloads and frame extraction (to queue as pair with frame number)
    """
    frame_queue = queue.Queue()

    sf = threading.Thread(target=get_stream_and_frames, args=(session["url"], frame_queue, target_fps, run_time,), daemon=True)
    sf.start()

    """
    Using Threads instead of Multiprocessing for now
    Make multiprocessing for running model on Frames (put into priorty queue sorted by frame number)
    The multiprocessing isn't too important as it is not time limiting on GTX 1080
    On weaker computer it may cause the video processing to run behind
    """
    processed_queue = queue.PriorityQueue()
    rm = threading.Thread(target=run_model_on_queue_loop, args=(frame_queue, processed_queue, target_fps, run_time,), daemon=True)
    rm.start()

    return Response(gen_frames(processed_queue, target_fps, run_time), mimetype='multipart/x-mixed-replace; boundary=frame')


"""
Based on code from https://blog.miguelgrinberg.com/post/video-streaming-with-flask
"""
def gen_frames(processed_queue, target_fps, run_time = 300):  # generate frame by frame from camera

    """
    Initialize first video segment and camera feed
    """

    start_time = time.time()

    last_frame_shown_time = time.time()
    while True:
        #forces function to exit after set time
        if time.time() - start_time > run_time and processed_queue.empty() is True:
            exit()

        count, frame = processed_queue.get()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        while(time.time() - last_frame_shown_time < (1 / target_fps)):
            time.sleep((1/(target_fps*4)))

        last_frame_shown_time = time.time()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/api/analyze_recorded', methods=['GET'])
def api_analyze_recorded():
    if 'path' in request.args:
        video_path = request.args.get('path')
    else:
        return IOError("Missing path in argument")

    print("API video path is:", video_path)

    num_directions = 8
    if 'num_directions' in request.args:
        num_directions = int(request.args.get('num_directions'))

    # save extracted data as file path without extensions and slashes
    video_name = Path(video_path).stem
    
    data_save_name =video_name.__str__()
    print("API video name is:", data_save_name)

    frame_queue = queue.Queue()
    processed_queue = queue.Queue()
    image_extraction_to_queue(video_path.__str__(), frame_queue, image_per_second=20, save_images = "off")
    model = load_saved_model()
    ct = CentroidTracker(save_file_name=data_save_name)
    run_model_on_queue(model, ct, frame_queue, processed_queue ,fps = 20)

    data_save_name = "last30_" + data_save_name + ".txt"

    json_path = Path.cwd() / "app" / "static" / "reports" / data_save_name

    print(json_path.__str__())
    
    # analyze 30 second reports from system
    return convert_log_to_json_summary(json_path.__str__(), num_directions)


@app.route('/api/get_summary/<video_name>', methods=['GET'])
def api_get_summary(video_name):
    print("Video Name passed to API is:", video_name)

    file_name = "last30_" + video_name + ".txt"

    file_path = Path.cwd() / 'app' / 'static' / "reports" / file_name

    return convert_log_to_json(file_path.__str__())

def convert_log_to_json(filename):
    with open(filename) as reports_30_sec_file:
        time_dict = json.loads(reports_30_sec_file.read())
    return time_dict

    


def convert_log_to_json_summary(filename, num_directions = 8):
    with open(filename) as reports_30_sec_file:
        time_dict = json.loads(reports_30_sec_file.read())
    time_list = [] 

    # 8 possible directions for cars to be counted in
    possible_directions = list(card_directions.keys())

    vehicle_counts_30 = []
    vehicle_avgspeed_30 = []

    _count_keyword = 'count'
    _avgspeed_keyword = 'average_speed'

    for t, v in time_dict.items():
        time_list.append(int(t))
        
        curr_vehicle_count = []
        curr_vehicle_avgspeed = []

        i_direction = 0
        for direction, info in v.items():
            if direction != possible_directions[i_direction]:
                curr_vehicle_count.append(0)
                curr_vehicle_avgspeed.append(0)
            i_direction += 1 

            curr_vehicle_count.append(int(info[_count_keyword]))
            curr_vehicle_avgspeed.append(int(info[_avgspeed_keyword]))
        
        vehicle_counts_30.append(curr_vehicle_count)
        vehicle_avgspeed_30.append(curr_vehicle_avgspeed)
    
    vehicle_counts_30 = np.array(vehicle_counts_30)
    vehicle_avgspeed_30 = np.array(vehicle_avgspeed_30)

    total_counts = np.sum(vehicle_counts_30, axis=0)
    direction_indices_used = total_counts.argsort()[-num_directions:][::-1]

    possible_directions =  [possible_directions[i] for i in direction_indices_used]
    vehicle_counts_30 = np.array([vehicle_counts_30[:, i] for i in direction_indices_used])
    vehicle_avgspeed_30 = np.array([vehicle_avgspeed_30[:, i] for i in direction_indices_used])

    vehicle_flow_per_min = (vehicle_counts_30 * 2)
    avg_vehicle_flow_per_min = np.mean(vehicle_flow_per_min, axis=1)

    normalized_avgspeed_30 = []
    max_avgspeeds = list(vehicle_avgspeed_30.max(axis=1))
    for i_dir in range(num_directions):
        normalized_avgspeed_30.append(np.around(vehicle_avgspeed_30[i_dir,:] / max_avgspeeds[i_dir], decimals=4))

    ret_dict = {}
    summary_dict = {}
    summary_dict["number_direction_analyzed"] = num_directions
    summary_dict["directions_of_flow"] = possible_directions
    summary_dict["direction_of_highest_avg_flow"] = \
        possible_directions[np.argmax(avg_vehicle_flow_per_min,axis=0)]
    summary_dict["avg_flow_per_minute"] = avg_vehicle_flow_per_min
    
    ret_dict["summary"] = summary_dict
    ret_dict["segment_end_times_in_seconds"] = time_list

    flow_per_minute_by_30_seconds = {}
    for i_dir in range(len(possible_directions)):
        flow_per_minute_by_30_seconds[possible_directions[i_dir]] = \
            list(vehicle_flow_per_min[i_dir, :])
    ret_dict["flow_per_minute_by_30_seconds"] = flow_per_minute_by_30_seconds

    relative_speeds_by_30_seconds = {}
    for i_dir in range(len(possible_directions)):
        relative_speeds_by_30_seconds[possible_directions[i_dir]] = \
            list(normalized_avgspeed_30[i_dir])
    ret_dict["relative_speeds_by_30_seconds"] = relative_speeds_by_30_seconds        

    return json.dumps(ret_dict, cls=NpEncoder)
        
if __name__ == '__main__':
    app.run(debug=True)