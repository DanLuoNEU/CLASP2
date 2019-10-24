# This file is used to generate hand and elbow visualization 
# 09/23/2019 Dan Luo
import os
import cv2
import json
import numpy as np
import scipy.io as sio
from progress.bar import Bar
from numpy.core.records import fromarrays

###### Draw hands on Bin visualization ######
# # Load all needed joints information produced by openpose into one dictionary
# img_root = '/home/ubuntu/Demo/CLASP-Project/Person-bin Detection/faster_rcnn_pytorch/examples/cam09exp2'
# img_list = os.listdir(img_root)
# img_list.sort()
# json_root = img_root+'_output/'
# json_list = os.listdir(json_root)
# json_list.sort()
# persons_joints = { 'image_name':[],
#                    'people':[]}
# joints_index = [ 4, 7, 3, 6 ]

# img_id = 0
# for json_file in json_list:
#     with open(json_root+json_file) as f:
#         people = json.load(f)['people']
#         persons_joints['image_name'].append(img_root+'/'+img_list[img_id])
#         people_tmp = []
#         for i in range(len(people)):
#             # Original joint information dimension is (3(x,y,confidence) x 25(num_joints))
#             # {4,  "RWrist"}, {7,  "LWrist"}, {3,  "RElbow"}, {6,  "LElbow"}
#             joints_tmp = np.array(people[i]['pose_keypoints_2d']).reshape(25,3)
#             people_tmp.append(joints_tmp[joints_index])
#
#         persons_joints['people'].append(people_tmp)
#     img_id += 1
#
# # Process and Store the hand detection visualization
# with Bar('Processing joint images', max=len(img_list)) as bar:
#     for img_id in range(len(img_list)):
#         img = cv2.imread(img_root+'/'+img_list[img_id])
#         for person in persons_joints['people'][img_id]:
#             # print(person[0,0],person[0,1])
#             # exit(0)
#             cv2.circle(img, (int(person[0,0]),int(person[0,1])), 10, (0,0,255),-1)
#             cv2.circle(img, (int(person[1,0]),int(person[1,1])), 10, (0,0,255),-1)
#         cv2.imwrite('cam09exp2/{}.jpg'.format(img_id),img)
#         bar.next()
    

###### Generate hands.json for specific clip ######
# Load all needed joints information produced by openpose into one dictionary
img_root = '/home/ubuntu/Demo/CLASP-Project/Person-bin Detection/faster_rcnn_pytorch/examples/cam09exp2'
img_list = os.listdir(img_root)
img_list.sort()
json_root = img_root+'_output/'
json_list = os.listdir(json_root)
json_list.sort()
hands = { 'frame_id':[], 'hands':[]}
joints_index = [ 4, 7, 3, 6 ]

for json_file in json_list:
    frame_id = int(json_file.split('_')[0])
    with open(json_root+json_file) as f:
        people = json.load(f)['people']
        hands['frame_id'].append(frame_id)
        hands_tmp = []
        for i in range(len(people)):
            # Original joint information dimension is (25(num_joints) x 3(x,y,confidence))
            joints_tmp = np.array(people[i]['pose_keypoints_2d']).reshape(25,3)
            hands_tmp.append(joints_tmp[joints_index])

        hands['hands'].append(hands_tmp)

sio.savemat('data/cam09exp2_hands.mat', hands)

# with open('data/cam09exp2_hands.json','w') as outfile:
#     json.dump(hands, outfile)

print('well done!')