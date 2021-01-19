"""
Dataset sourced from: STREETS
https://databank.illinois.edu/datasets/IDB-3671567
"""

import json
import os
import re
import shutil
import cv2
import numpy as np

json_file_name = 'annotations/vehicle-annotations.json'
# json_file_name = 'annotations/vehicle-annotations-1-photo.json' # for testing
img_dimensions = [480, 720]

file_name_pretty = json_file_name[:-5] + '-pretty.json'
mask_path = 'mask'
images_path = 'images'

def line_readline(f):
    line = f.readline()
    if line == '':
        return line
    line = line.strip()
    line = line.replace('{', '')
    line = line.replace('}', '')
    line = line.replace('[', '')
    line = line.replace(']', '')
    if line == '':
        return ''
    line_strings = re.findall(r'"(.*?)"', line)
    if len(line_strings) > 1:
        return line_strings
    if len(line_strings) > 0:
        return line_strings[0]
    line = line.replace(',', '')
    return line

line_counter = 1
def read_non_empty_line(f, f_len):
    global line_counter
    if line_counter > f_len:
        return ''

    line = line_readline(f)
    line_counter += 1

    while line == '':
        if line_counter > f_len:
            return ''
        line = line_readline(f)
        line_counter += 1
    return line

def main():
    # read json file for annotations
    print('reading file from ', json_file_name)
    with open(json_file_name) as va:
        data = json.load(va)

    # write pretty copy of json file
    print('writing prettify json to ', file_name_pretty)
    with open(json_file_name[:-5] + '-pretty.json', 'w') as va_pretty:
        va_pretty.write(json.dumps(data, indent=0))

    # recreate mask folder for new mask
    path = os.getcwd()
    mask_filepath = '{}/{}'.format(path, mask_path)
    try:
        shutil.rmtree(mask_filepath)
        os.mkdir(mask_filepath)
    except OSError:
        print('Failed to create mask folder')
    print('Created folder for mask images ', mask_filepath)

    # read pretty json and create mask images
    f_data_len = len(open(file_name_pretty, 'r').readlines())
    f_data = open(file_name_pretty, 'r')
    print('------------------------------------------------------------------')

    """
    steps 
    0 obj_name
    1 fileref
    2 size
    3 filename
    4 base64_img_data
    5 file_attributes
    6 regions
    7   region number
    8   shape attributes
    9   all_points_x
    10  all_points_y
    11  shape name
    12  region_attributes
    13  vehicle_type

    """
    steps = ['', 'fileref', 'size', 'filename', 'base64_img_data', 
        'file_attributes', 'regions', '', 'shape_attributes', 'all_points_x', 
        'all_points_y', 'name', 'region_attributes', 'vehicle_type']

    num_masks_created = 0
    non_cars_added = 0

    line = read_non_empty_line(f_data, f_data_len)

    # repeat for all images
    while line != '' and line_counter < f_data_len:
        for _ in range(2):
            read_non_empty_line(f_data, f_data_len)
        name_filename = read_non_empty_line(f_data, f_data_len)
        if isinstance(name_filename, list) and name_filename[0] == steps[3]:
            mask_filename = name_filename[1]
        else:
            print("ERROR: could not find filename for mask")
            return 0

        for _ in range(2):
            read_non_empty_line(f_data, f_data_len)
        name_regions = read_non_empty_line(f_data, f_data_len)
        if not name_regions == steps[6]:
            print("ERROR: could not find regions for mask ", mask_filename)
            line = read_non_empty_line(f_data, f_data_len)
            continue

        # start adding points
        polygons_list = []
        next_line = read_non_empty_line(f_data, f_data_len)
        while next_line:
            if not next_line.isnumeric():
                break

            shape_attributes = read_non_empty_line(f_data, f_data_len)
            if not shape_attributes == steps[8]:
                print("ERROR: could not find shape attributes for mask ", mask_filename)
                line = read_non_empty_line(f_data, f_data_len)
                continue

            points_list = []
            name_all_points_x = read_non_empty_line(f_data, f_data_len)
            next_line = read_non_empty_line(f_data, f_data_len)
            while next_line.replace('-','').isnumeric():
                points_list.append([next_line])
                next_line = read_non_empty_line(f_data, f_data_len)
            name_all_points_y = next_line

            point_counter = 0
            next_line = read_non_empty_line(f_data, f_data_len)
            while not isinstance(next_line, list) and next_line.replace('-','').isnumeric():
                points_list[point_counter].append(next_line)
                next_line = read_non_empty_line(f_data, f_data_len)
                point_counter += 1

            name_name = next_line
            region_attributes = read_non_empty_line(f_data, f_data_len)
            vehicle_type = read_non_empty_line(f_data, f_data_len)

            if isinstance(vehicle_type, list) and vehicle_type[0] == steps[13]:
                if vehicle_type[1] != 'car':
                    non_cars_added += 1

            polygons_list.append(points_list)
            next_line = read_non_empty_line(f_data, f_data_len)

        # create image
        image_filepath = '{}/{}'.format(images_path, mask_filename)
        img_h, img_w = cv2.imread(image_filepath).shape[:2]

        mask_img = np.zeros((img_h, img_w))
        for polygon in polygons_list: 
            polygon = np.array(polygon, dtype=int)
            cv2.fillPoly(mask_img, [polygon], color=(255,255,255))
        mask_img_filepath = '{}/{}'.format(mask_path, mask_filename)
        cv2.imwrite(mask_img_filepath, mask_img)
        num_masks_created += 1
        print('wrote image #{} to {}'.format(num_masks_created, mask_img_filepath) , end='\t\t\t\r')

    f_data.close()
    print('\nAdded {} mask images to mask folder'.format(num_masks_created))
    print('Total of {} non-car vehicles in this dataset'.format(non_cars_added))

if __name__ == '__main__':
    main()