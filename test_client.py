import requests
import json

# TEST 1: Check API can connect and receive successfully
resp = requests.get('http://127.0.0.2:5000/api/test')
if resp.status_code != 200:
    raise OSError('Error when testing API {}'.format(resp.status_code))
resp_dict = json.loads(resp.content)
print('Test Content: ', resp_dict)
print('Test Response converted to ', type(resp_dict))
# check success is returned
assert('success' in resp_dict.keys())

# TEST 2: Check API can accept highway video and provide traffic flow data
parameters = {
    "path": "C:\\Users\\AndrewLaptop\\hwy-cam-1189--11.mp4",
    "num_directions": 2
}
resp = requests.get('http://127.0.0.2:5000/api/analyze_recorded', params=parameters)
if resp.status_code != 200:
    raise OSError('Error when analyzing video {}'.format(resp.status_code))
resp_dict = json.loads(resp.content)
print(resp_dict)
