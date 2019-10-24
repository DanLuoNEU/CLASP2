# This file is used to generate .mat file from the openpose json output file
# 09/23/2019 Dan Luo
import os
import json
import numpy as np
import scipy.io as sio
from numpy.core.records import fromarrays
from progress.bar import Bar

# Load all needed joints information produced by openpose into one dictionary
img_root = '/home/ubuntu/Demo/CLASP-Project/Person-bin Detection/faster_rcnn_pytorch/examples/cam09exp2'
img_list = os.listdir(img_root)
img_list.sort()
json_root = img_root+'_output/'
json_list = os.listdir(json_root)
json_list.sort()
persons_joints = { 'image_name':[],
                   'people':[]}
joints_index = [ 4, 7, 3, 6 ]

bar = Bar('Processing Joints', max=len(json_list))
img_id = 0
for json_file in json_list:
    with open(json_root+json_file) as f:
        people = json.load(f)['people']
        persons_joints['image_name'].append(img_root+'/'+img_list[img_id])
        people_tmp = []
        for i in range(len(people)):
            # Original joint information dimension is (3(x,y,confidence) x 25(num_joints))
            # {4,  "RWrist"}, {7,  "LWrist"}, {3,  "RElbow"}, {6,  "LElbow"}
            joints_tmp = np.array(people[i]['pose_keypoints_2d']).reshape(25,3)
            people_tmp.append(joints_tmp[joints_index])
            # people_tmp.append(joints_tmp)

        persons_joints['people'].append(people_tmp)
    img_id += 1
    bar.next()
bar.finish()


# Transform the Dictionary into .mat file that clasp1 needs ,persons_joints['sub']
sio.savemat('cam09exp2_joints_all.mat', persons_joints)

print('well done!')


