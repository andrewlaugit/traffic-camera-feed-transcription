"""
Dataset sourced from: STREETS
https://databank.illinois.edu/datasets/IDB-3671567
"""

import os
import os.path

import re
import shutil
import cv2

"""
all samples have same format - <streetname>-<0-9>.jpg.
Will try to check if sample for roadmask exists, and create
duplicate image if does.
"""

num_masks_added = 0

images_filepath = 'images/'
roadmask_dest_filepath = 'roadmasks/'
roadmask_orig_filepath = 'roadmasks_originals/'

filepaths = os.listdir(roadmask_orig_filepath)

# recreate roadmask folder for new masks
path = os.getcwd()
mask_filepath = '{}/{}'.format(path, "roadmasks")
try:
    shutil.rmtree(mask_filepath)
    os.mkdir(mask_filepath)
except OSError:
    print('Failed to create road mask folder')
print('Created folder for road mask images ', mask_filepath)


for full_file_name in filepaths:
    file_name = full_file_name[:-4]
    roadmask = cv2.imread(roadmask_orig_filepath + full_file_name)
    roadmask[roadmask != 0] = 255

    for i in range(10):
        if os.path.exists('{}{}-{}.jpg'.format(images_filepath, file_name, i)):
            num_masks_added += 1
            print('wrote image #{} for {}'.format(num_masks_added, file_name) , end='\t\t\t\r')
            cv2.imwrite('{}{}-{}.jpg'.format(roadmask_dest_filepath, file_name, i), roadmask)

print()
print(num_masks_added)