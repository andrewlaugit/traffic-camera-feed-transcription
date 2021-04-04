import requests
import json
import time
import csv


def convert_to_csv(resp_dict, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        directions = resp_dict['summary']['directions_of_flow']
        writer.writerow(['time'] + directions + directions)

        time_intervals = resp_dict['segment_end_times_in_seconds']
        car_counts = []
        rel_speed = []
        for d in directions:
            car_counts.append(resp_dict['flow_per_minute_by_30_seconds'][d])
            rel_speed.append(resp_dict['relative_speeds_by_30_seconds'][d])

        for i in range(len(time_intervals)):
            new_row = [time_intervals[i], ]
            for d in car_counts:
                new_row.append(d[i]/2)
            for rs in rel_speed:
                new_row.append(rs[i])

            writer.writerow(new_row)

# TEST 1: Check API can connect and receive successfully
resp = requests.get('http://127.0.0.1:5000/api/test')
if resp.status_code != 200:
    raise OSError('Error when testing API {}'.format(resp.status_code))
resp_dict = json.loads(resp.content)
print('Test Content: ', resp_dict)
print('Test Response converted to ', type(resp_dict))
# check success is returned
assert('success' in resp_dict.keys())

# TEST 2: Check API can accept recorded highway video and provide traffic flow data
paremeters_list = [
    {
        "path": "C:\\Users\\Bob\\Desktop\\traffic-camera-feed-transcription\\app\\static\\R11_011_2021_03_21_19-32-46.mp4",
        "num_directions": 2
    # },
    # {
    #     "path": "C:\\Users\\Bob\\Desktop\\traffic-camera-feed-transcription\\test_videos\\hwy-cam-1189--11.mp4",
    #     "num_directions": 2
    # },
    # {
    #     "path": "C:\\Users\\Bob\\Desktop\\traffic-camera-feed-transcription\\test_videos\\hwy-cam-1208--11.mp4",
    #     "num_directions": 2
    # },
    # {
    #     "path": "C:\\Users\\Bob\\Desktop\\traffic-camera-feed-transcription\\test_videos\\hwy-cam-3356--4.mp4",
    #     "num_directions": 2
    # },
    # {
    #     "path": "C:\\Users\\Bob\\Desktop\\traffic-camera-feed-transcription\\test_videos\\hwy-camera-944--9.mp4",
    #     "num_directions": 2
    # },
    # {
    #     "path": "C:\\Users\\Bob\\Desktop\\traffic-camera-feed-transcription\\test_videos\\hwy-I-95 at Middletown Road.mp4",
    #     "num_directions": 2
    # },
    # {
    #     "path": "C:\\Users\\Bob\\Desktop\\traffic-camera-feed-transcription\\test_videos\\int-camera-46--7.mp4",
    #     "num_directions": 4
    # },
    # {
    #     "path": "C:\\Users\\Bob\\Desktop\\traffic-camera-feed-transcription\\test_videos\\int-camera-70--7.mp4",
    #     "num_directions": 4
    # },
    # {
    #     "path": "C:\\Users\\Bob\\Desktop\\traffic-camera-feed-transcription\\test_videos\\int-camera-125--7.mp4",
    #     "num_directions": 4
    # },
    # {
    #     "path": "C:\\Users\\Bob\\Desktop\\traffic-camera-feed-transcription\\test_videos\\int-camera-683--11.m4v",
    #     "num_directions": 8
    # },
    # {
    #     "path": "C:\\Users\\Bob\\Desktop\\traffic-camera-feed-transcription\\test_videos\\BandW_hwy-camera-944--9 (1).mp4",
    #     "num_directions": 2
    # # },
    # {
    #     "path": "C:\\Users\\Bob\\Desktop\\traffic-camera-feed-transcription\\test_videos\\int-NY Routes 5, 8, 12_Burrstone Road Interchange- Utica.mp4",
    #     "num_directions": 4
    }
]

computation_times = []
for parameters in paremeters_list:
    start = time.time()
    resp = requests.get('http://127.0.0.1:5000/api/analyze_recorded', params=parameters)
    if resp.status_code != 200:
        raise OSError('Error when analyzing video {}'.format(resp.status_code))
    end = time.time()
    computation_times.append(round(end-start, 4))
    print("Time for response: {:.4f}".format(computation_times[-1]))
    resp_dict = json.loads(resp.content)

    csv_save_name = 'csv\\' + ''.join(filter(str.isalnum, parameters['path'])) + '.csv'
    convert_to_csv(resp_dict, csv_save_name)
    print('Saved csv successfully')

print(computation_times)

# TEST 3: Check API can accept live video and provide traffic flow data
parameters = {
    "live_url": "https://35-d2.divas.cloud/CHAN-351/CHAN-351_1.stream/chunklist_w1962356171.m3u8",
    "num_directions": 2,
    "run_time": 20,
    "fps": 20
}
resp = requests.get('http://127.0.0.1:5000/api/analyze_stream', params=parameters)
if resp.status_code != 200:
    raise OSError('Error when analyzing live video {}'.format(resp.status_code))
resp_dict = json.loads(resp.content)
print(resp_dict)

