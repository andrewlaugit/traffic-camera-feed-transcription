import flask
from flask import request, jsonify

from werkzeug.utils import secure_filename
from pathlib import Path
import json
import numpy as np
import queue
import threading
from imageextraction import *
from car_detection import *
from live_stream import *
from centroidtracker import *

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Available Endpoints</h1>
        <p>analyze_recorded: path, num_directions<\p>'''

@app.route('/api/test', methods=['GET'])
def api_test():
    diction = {'success': True}
    return jsonify(diction)

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

@app.route('/api/analyze_recorded', methods=['GET'])
def api_analyze_recorded():
    if 'path' in request.args:
        video_path = request.args.get('path')
    else:
        return IOError("Missing path in argument")

    num_directions = 4
    if 'num_directions' in request.args:
        num_directions = int(request.args.get('num_directions'))
    else:
        return IOError("Missing number of directions in argument") 

    # save extracted data as file path without extensions and slashes
    data_save_name = ''.join(filter(str.isalnum, video_path)) + '.txt'
    # video_name = ''.join(filter(str.isalnum, video_path))

    frame_queue = queue.Queue()
    # processed_queue = queue.Queue()
    image_extraction_to_queue(video_path.__str__(), frame_queue, image_per_second=20, save_images = "off")
    model = load_saved_model()
    ct = CentroidTracker(file_name=data_save_name)
    # run_model_on_queue(model, ct, frame_queue, processed_queue,fps = 20)
    run_model_on_queue(model, ct, frame_queue, fps = 20)
    # make_video_from_queue(video_name, processed_queue, fps=20)

    # analyze 30 second reports from system
    return convert_log_to_json_summary('temp\\last30{}'.format(data_save_name), num_directions)
    # return convert_log_to_json_summary('temp\\last30_test.txt', num_directions)

'''
this code may not work!!!!!!!!!!!
'''
# @app.route('/api/analyze_live', methods=['GET'])
# def api_analyze_livestream():
#     if 'live_url' in request.args:
#         url = request.args.get('live_url')
#     else:
#         return IOError("Missing livestream url in argument")

#     num_directions = 4
#     if 'num_directions' in request.args:
#         num_directions = int(request.args.get('num_directions'))
#     else:
#         return IOError("Missing number of directions in argument") 

#     analysis_time = 300
#     if 'analysis_time' in request.args:
#         num_directions = int(request.args.get('analysis_time'))

#     target_fps = 20

#     """
#     Make single thread for segment downloads and frame extraction (to queue as pair with frame number)
#     """
#     frame_queue = queue.Queue()
   
#     sf = threading.Thread(target=get_stream_and_frames, args=(url, frame_queue, \
#                         target_fps, ), daemon=True)
#     sf.start()

#     """
#     Using Threads instead of Multiprocessing for now
#     Make multiprocessing for running model on Frames (put into priorty queue sorted by frame number)
#     The multiprocessing isn't too important as it is not time limiting on GTX 1080
#     On weaker computer it may cause the video processing to run behind
#     """
#     processed_queue = queue.PriorityQueue()
#     data_save_name = 'temp\\live_' + ''.join(filter(str.isalnum, url)) + '.txt'

#     rm = threading.Thread(target=run_model_on_queue_loop, args=(frame_queue, processed_queue, target_fps, data_save_name, ), daemon=True)
#     rm.start()
#     return convert_log_to_json_summary('temp\\last30_test.txt', num_directions)

def convert_log_to_json_summary(filename, num_directions):
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

app.run(host='127.0.0.2', port='5000')