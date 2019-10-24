import os
import cv2
import numpy as np
import scipy.io as sio
from progress.bar import Bar

img_path = 'cam09exp2_bj/'
save_path = 'cam09exp2_bjp/'
# frame, id, x1, y1, x2,y2
mat = sio.loadmat('peopleCam09.mat')
frame = mat['struct'][0][0]['frame'][0]
pid = mat['struct'][0][0]['id']
x1 = mat['struct'][0][0]['x1'][0]
y1 = mat['struct'][0][0]['y1'][0]
x2 = mat['struct'][0][0]['x2'][0]
y2 = mat['struct'][0][0]['y2'][0]
bbox = np.asarray([x1,y1,x2,y2])

# with Bar('Preparing person images', max=len(frame)) as bar:
#     frame_last = 1
#     for frame_id in frame:
#         bar.next()
#         if frame_id == frame_last:
#             continue
#         img = cv2.imread(img_path+str(frame_id)+".jpg")
#         cv2.imwrite(save_path+str(frame_id)+'.jpg', img)
#         frame_last = frame_id
        
with Bar('Processing person images', max=len(frame)) as bar:
    for i in range(len(frame)):
        img = cv2.imread(save_path+str(frame[i])+".jpg")
        # imS = cv2.resize(img, (960, 540))

        x1 = int(bbox[0][i])
        y1 = int(bbox[1][i])
        x2 = int(bbox[2][i])
        y2 = int(bbox[3][i])
        
        img = cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,0),2)
        img = cv2.putText(img, str(pid[i]), (x1,y1+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 4)
        
        cv2.imwrite(save_path+str(frame[i])+'.jpg', img)
        bar.next()
        # cv2.imshow('img',img)
        # cv2.waitKey(15)

print('Well Done!')