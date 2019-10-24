import os
import csv
import cv2
import numpy as np
from progress.bar import Bar
file_path = 'binsbins.csv'

# last to run

with open(file_path, 'r') as f:
	# frame, id, x1,y1,x2,y2
	lines = f.readlines()
	dicti = {}
	for line in lines:
		splitted = line.split(',')
		# print(line)
		frame_num = int(splitted[0])
		x1 = int(splitted[2])
		y1 = int(splitted[3])
		x2 = int(splitted[4])
		y2 = int(splitted[5])
		
		if(frame_num in dicti.keys()):
			dicti[frame_num].append([x1,y1,x2,y2])
		else:
			dicti[frame_num] = []
			dicti[frame_num].append([x1,y1,x2,y2])

# print(dicti)
# Iterate and draw bounding boxes
bar = Bar('Processing bin images', max=len(dicti))
for frame in dicti.keys():
	# Path to iamges
	img = cv2.imread(os.path.join('cam09exp2/',str(frame)+".jpg"))
	for bbox in dicti[frame]:
		x1 = bbox[0]
		y1 = bbox[1]
		x2 = bbox[2]
		y2 = bbox[3]

		img = cv2.rectangle(img, (x1,y1),(x2,y2), (0,255,255),2)
		imS = cv2.resize(img, (960, 540))

	cv2.imwrite('cam09exp2_bj/'+str(frame)+'.jpg', img)
	bar.next()
	# cv2.imshow('img',imS)

	# cv2.waitKey(15)	
bar.finish()