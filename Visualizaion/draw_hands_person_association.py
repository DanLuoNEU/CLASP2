# Visualize the hands and person association
# Dan, 09/30/2019
import os
import cv2
import pickle
from progress.bar import Bar

img_root = '/home/ubuntu/Demo/CLASP-Project/Person-bin Detection/faster_rcnn_pytorch/examples/cam09exp2'
img_list = os.listdir(img_root)
img_list.sort()

# Load Person Detection result{'bbox':,
#                              'id':,
#                              'hands:, 
#                              'bins':
# Load hands and person association file 
with open('data/cam09exp2_people_with_hand_association.pkl','r') as f:
    persons = pickle.load(f)
       

with Bar('Processing hands and people association visualization',max=len(persons['id'])) as bar:
    for frame_id in persons['id'].keys():
        img = cv2.imread(img_root+'/'+img_list[frame_id])
        for person_i in range(len(persons['bbox'][frame_id])):
            bbox = persons['bbox'][frame_id][person_i]
            pid = persons['id'][frame_id][person_i]
            hands = persons['hands'][frame_id][person_i]
            # Draw person bbox
            cv2.rectangle(img, (bbox[0],bbox[1]),(bbox[2],bbox[3]), (255,0,0),2)
            cv2.putText(img, str(pid), (bbox[0],bbox[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 4)
            # Draw hands with id
            if hands == []:
                continue
            if (hands[0,0] != 0) or (hands[0,1] != 0):
                cv2.circle(img, (hands[0,0],hands[0,1]), 10, (0,0,255),-1)
                cv2.putText(img, str(pid), (hands[0,0],hands[0,1]+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4)
            if (hands[1,0] != 0) or (hands[1,1] != 0):
                cv2.circle(img, (hands[1,0],hands[1,1]), 10, (0,0,255),-1)
                cv2.putText(img, str(pid), (hands[1,0],hands[1,1]+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4)
        cv2.imwrite('cam09exp2_hands_person_association/{}.jpg'.format(frame_id),img)
        bar.next()

print("Well Done!")
