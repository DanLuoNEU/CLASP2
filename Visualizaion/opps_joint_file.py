# This file is used to generate .pkl/.mat/.json file from the openpose json output file
# 09/23/2019 Dan Luo
import os
import json
import pickle
import numpy as np
import scipy.io as sio
from progress.bar import Bar
from numpy.core.records import fromarrays

# Data would save in the form of 
# persons_joints: {dictionary
#                   frame, keys
#                   {dictionary
#                    ['image_name'],string of /path/to/image
#                    ['people'],list of joints
#                   }, values


# Load all needed joints information produced by openpose into one dictionary
# img_root = '/home/ubuntu/Demo/CLASP-DATA-102319/cam11exp2'
img_root = '/home/ubuntu/Demo/CLASP2/Visualizaion/CLASP-DATA-102319/cam13exp2'
# save_name = 'cam09exp2_joints_all.mat'
save_name = 'data/joints_all_cam13exp2_102419.pkl'
img_list = os.listdir(img_root)
img_list.sort()
# json_root = img_root+'_results/'
json_root = '/home/ubuntu/Demo/CLASP-DATA-102319/cam13exp2_results/'
json_list = os.listdir(json_root)
json_list.sort()
persons_joints = {}
# {4,  "RWrist"}, {7,  "LWrist"}, {3,  "RElbow"}, {6,  "LElbow"}
joints_index = [ 4, 7, 3, 6 ]

img_id = 0
bar = Bar('Processing Joints', max=len(json_list))
for json_file in json_list:
    with open(json_root+json_file) as f:
        frame = int(json_file.split('_')[0].split('frame')[1])
        persons_joints[frame]={}
        persons_joints[frame]['image_name'] = img_root+'/'+img_list[img_id]
        people = json.load(f)['people']
        persons_joints[frame]['people'] = []
        for i in range(len(people)):
            # Original joint information dimension is (3(x,y,confidence) x 25(num_joints))
            joints_tmp = np.array(people[i]['pose_keypoints_2d']).reshape(25,3)
            # Save specific arm information
            # people_tmp.append(joints_tmp[joints_index])
            # Save the whole skeleton
            persons_joints[frame]['people'].append(joints_tmp)
    
    img_id += 1
    bar.next()
bar.finish()


# Transform the Dictionary into .mat file that clasp1 needs ,persons_joints['sub']
# Save .mat file
# sio.savemat(save_name, persons_joints)
# Save .pkl file
with open(save_name, 'wb') as f:
    pickle.dump(persons_joints, f)

print('well done!')


