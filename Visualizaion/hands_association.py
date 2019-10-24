# Build up association between hands and persons ID, 
# depending on IOU between skeleton and person bounding boxes
# Intersection part reference: https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py#L48
# Dan, 09/29/2019

import os
import cv2
import json
import pickle
import numpy as np
import scipy.io as sio
from progress.bar import Bar
from numpy.core.records import fromarrays

person_path = '/home/ubuntu/Demo/CLASP-Project/Visualizaion/data/cam09exp2_logs_full_segv2.txt'
joints_path = '/home/ubuntu/Demo/CLASP-Project/Visualizaion/data/cam09exp2_joints_all.mat'
# Load Person Detection result{'bbox':,
#                              'id':,
#                              'bins':
#                             }
#  and Joint Result{'frame_id':
#                   'peoples': 
#                  }

def jaccard(box_a, boxes_b):
    """Compute the jaccard overlap of one box and a list of boxes.  
    The jaccard overlap is simply the intersection over union of two boxes.  
    Here we operate on person box and skeleton boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (list) Person bounding box, Shape: [4,]
        boxes_b: (list) Skeleton bounding boxes, Shape: [num_skeletons,4]
    Return:
        jaccard overlap: (tensor) Shape: [boxes_b.size(0)]
    """
    b_a = np.asarray(box_a)[np.newaxis,:]
    b_b = np.asarray(boxes_b)
    num_a = b_a.shape[0]
    num_b = b_b.shape[0]
    min_xy_a = np.repeat(np.expand_dims(b_a[:,:2], 1),num_b,axis=1)
    min_xy_b = np.repeat(np.expand_dims(b_b[:,:2], 0),num_a,axis=0)
    max_xy_a = np.repeat(np.expand_dims(b_a[:,2:], 1),num_b,axis=1)
    max_xy_b = np.repeat(np.expand_dims(b_b[:,2:], 0),num_a,axis=0)
    min_xy = np.maximum(min_xy_a, min_xy_b) 
    max_xy = np.minimum(max_xy_a, max_xy_b)
    
    inter_xy = np.clip((max_xy - min_xy), 0, np.inf)
    inter = inter_xy[:,:,0] * inter_xy[:,:,1]
    area_a = np.repeat(np.expand_dims(((b_a[:, 2]-b_a[:, 0]) * (b_a[:, 3]-b_a[:, 1])), 1),num_b,axis=1)
    area_b = np.repeat(np.expand_dims(((b_b[:, 2]-b_b[:, 0]) * (b_b[:, 3]-b_b[:, 1])), 0),num_a,axis=0)
    union = area_a + area_b - inter

    return (inter/union)[0,:]

# Load persons data
with open(person_path, 'r') as f:
	# frame, id, x1,y1,x2,y2
    lines = f.readlines()
    persons = {'id':{}, 'bbox':{},'bins':{},'hands':{}}
    for line in lines:
        splitted = line.split(',')
        frame_num = int(splitted[0])
        pid = splitted[1]
        x1 = int(splitted[2])
        y1 = int(splitted[3])
        x2 = int(splitted[4])
        y2 = int(splitted[5])

        if(frame_num not in persons['id'].keys()):
            persons['id'][frame_num] = []
            persons['bbox'][frame_num] = []
            persons['hands'][frame_num] = []
            persons['bins'][frame_num] = []

        persons['id'][frame_num].append(pid)
        persons['bbox'][frame_num].append([x1,y1,x2,y2])
        persons['hands'][frame_num].append([])
        persons['bins'][frame_num].append([])

# Load joints estimation results
joints_mat = sio.loadmat(joints_path)
skeletons = joints_mat['people'][0]

# For every frame, for every person bbox, for every skeleton
# compute IOU between person bbox and skeleton bbox
# Attach hands info to persons data
bar = Bar('Processing hands association:', max=len(persons['id']))
for frame_id in persons['id'].keys():
    # Build bounding box for each skeleton
    # REMEMBER to filter the (0,0) joints
    bboxes_skeleton = []
    for skeleton in skeletons[frame_id]:
        ## Avoid that (0,0) point is always the top left point
        for joint in skeleton:
            if joint[0] != 0 and joint[1] != 0:
                x_min, x_max = joint[0],joint[0] 
                y_min, y_max = joint[1],joint[1]
        for joint in skeleton:
            if joint[0] != 0 and joint[1] != 0:
                if joint[0] < x_min: x_min = joint[0]
                elif joint[0] > x_max: x_max = joint[0]
                    
                if joint[1] < y_min: y_min = joint[1]
                elif joint[1] > y_max: y_max = joint[1]
        bboxes_skeleton.append([int(x_min), int(y_min), int(x_max), int(y_max)])
    # Find the skeleton with largest IOU with the person bounding box
    for ind in range(len(persons['bbox'][frame_id])):
        bbox = persons['bbox'][frame_id][ind]
        # compute IOU
        IOUs = jaccard(bbox, bboxes_skeleton)
        skeleton_id = np.argmax(IOUs)
        if IOUs[skeleton_id] != 0:
            persons['hands'][frame_id][ind]=(skeletons[frame_id][skeleton_id][[4,7]]).astype(int)
    bar.next()
bar.finish()

# # Test if hands in person bounding box 
# for frame_id in persons['id'].keys():
#     print(persons['hands'][frame_id],persons['bbox'][frame_id]) 
                     
# Save using pickle, success
with open('data/cam09exp2_people_with_hand_association.pkl','wb') as f:
    pickle.dump(persons, f)

# # Test if save the right file 
# with open('data/cam09exp2_people_with_hand_association.pkl','r') as f:
#     persons = pickle.load(f)        

print("Well Done!")