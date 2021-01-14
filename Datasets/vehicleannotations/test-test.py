
import json

json_file_name = 'annotations/vehicle-annotations-1-photo.json'

# read json file for annotations
print('reading file from ', json_file_name)
with open(json_file_name) as va:
    data = json.load(va)

print(data.values())