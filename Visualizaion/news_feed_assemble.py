import os
import cv2
import csv
import json
import pickle
import numpy as np
import scipy.io as sio
from progress.bar import Bar
from numpy.core.records import fromarrays

frame_info = {'frame_id':{
                        'persons':{
                            'bbox':[],
                            'id':[]},
                        'bins':{ 
                            'bbox':[],
                            'id':[],
                            'owner':[]},
                        'hands':[]
                        }
}

bin_path = '/home/ubuntu/Demo/CLASP-Project/Visualizaion/data/cam09exp2_bins_RPI_0926.csv'
person_path = '/home/ubuntu/Demo/CLASP-Project/Visualizaion/data/peopleCam09.mat'
hands_path = '/home/ubuntu/Demo/CLASP-Project/Visualizaion/data/cam09exp2_hands.mat'

# Load Bin results
with open(bin_path, 'r') as f:
	# frame, id, x1,y1,x2,y2
    lines = f.readlines()
    bins = {}
    for line in lines:
        splitted = line.split(',')
        frame_num = int(splitted[0])
        bid = int(splitted[1])
        x1 = int(splitted[2])
        y1 = int(splitted[3])
        x2 = int(splitted[4])
        y2 = int(splitted[5])
		
        if(frame_num in bins.keys()):
            bins[frame_num]['id'].append(bid)
            bins[frame_num]['bbox'].append([x1,y1,x2,y2])
            bins[frame_num]['owner'].append([])
        else:
            bins[frame_num] = {'id':[], 'bbox':[],'owner':[]}
            bins[frame_num]['id'].append(bid)
            bins[frame_num]['bbox'].append([x1,y1,x2,y2])
            bins[frame_num]['owner'].append([])

# Load Person results
# frame, id, x1, y1, x2,y2
mat = sio.loadmat(person_path)
frame_t = mat['struct'][0][0]['frame'][0]
pid_t = mat['struct'][0][0]['id']
x1 = mat['struct'][0][0]['x1'][0]
y1 = mat['struct'][0][0]['y1'][0]
x2 = mat['struct'][0][0]['x2'][0]
y2 = mat['struct'][0][0]['y2'][0]
bbox_t = np.asarray([x1,y1,x2,y2])

people = {}
for i in range(len(frame_t)):
    p_bbox = [int(x1[i]),int(y1[i]),int(x2[i]),int(y2[i])]
    if(frame_t[i] in people.keys()):
        people[frame_t[i]]['id'].append(int(pid_t[i]))
        people[frame_t[i]]['bbox'].append(p_bbox)
    else:
        people[frame_t[i]] = {'id':[], 'bbox':[]}
        people[frame_t[i]]['id'].append(int(pid_t[i]))
        people[frame_t[i]]['bbox'].append(p_bbox)

# Load Hands results
hands_mat = sio.loadmat(hands_path)
frame_t = hands_mat['frame_id'][0]
hands_t = hands_mat['hands'][0]
hands = {}
for i in range(len(hands_t)):
    hands[frame_t[i]]=hands_t[i]

# All for One
news_feed = {'people':people, 'bins':bins, 'hands':hands}
### mat saving is not working
# sio.savemat('data/news_feed_before.mat', news_feed)
# m_t = sio.loadmat('data/news_feed_before.mat')

## Save using pickle, success
output_file = open('data/news_feed_before.pkl','wb')
pickle.dump(news_feed, output_file)
output_file.close()

print('Well Done!')