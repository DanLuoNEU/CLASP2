# This file is used to generate joints visualization 
# 09/23/2019 Dan Luo
import os
import cv2
import json
import numpy as np
import scipy.io as sio
from progress.bar import Bar
from numpy.core.records import fromarrays

# ############### Draw Joints on bin and person visualization ########################
# Load all needed joints information produced by openpose into one dictionary
img_root = '/home/ubuntu/Demo/CLASP-Project/Person-bin Detection/faster_rcnn_pytorch/examples/cam09exp2'
img_list = os.listdir(img_root)
img_list.sort()
json_root = img_root+'_output/'
json_list = os.listdir(json_root)
json_list.sort()
persons_joints = { 'image_name':[], 'people':[]}
joints_index = [ 4, 7, 3, 6 ]

img_id = 0
for json_file in json_list:
    with open(json_root+json_file) as f:
        people = json.load(f)['people']
        persons_joints['image_name'].append(img_root+'/'+img_list[img_id])
        people_tmp = []
        for i in range(len(people)):
            # Original joint information dimension is (25(num_joints) x 3(x,y,confidence))
            joints_tmp = np.array(people[i]['pose_keypoints_2d']).reshape(25,3)
            people_tmp.append(joints_tmp)

        persons_joints['people'].append(people_tmp)
    img_id += 1

# Get all specific image ids from clips
bjp_img_root = '/home/ubuntu/Demo/CLASP-Project/Visualizaion/cam09exp2_bjp'
bjp_img_list = os.listdir(bjp_img_root)
bjp_img_list.sort()
bjp_img_id = []
for img_name in bjp_img_list:
    bjp_img_id.append(int(img_name.split('.')[0]))

# Process and Store the hand detection visualization
with Bar('Processing joint images', max=len(bjp_img_id)) as bar:
    for img_id in range(len(bjp_img_list)):
        img = cv2.imread(bjp_img_root+'/'+bjp_img_list[img_id])
        for person in persons_joints['people'][bjp_img_id[img_id]]:
            # print(person[0,0],person[0,1])
            # exit(0)
            for joint in person:
                cv2.circle(img, (int(joint[0]),int(joint[1])), 10, (0,0,255),-1)
        # cv2.imwrite('cam09exp2_bjpj/{}.jpg'.format(bjp_img_id[img_id]),img)
        # cv2.imshow('img',img)
        # cv2.waitKey(15)
        bar.next()
    
print('well done!')