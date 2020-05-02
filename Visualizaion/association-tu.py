########## Import Part ##########
import os
import sys
import cv2
import csv
import time
import pickle
import operator
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
########## Configuration ##########
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (900,50)
fontScale              = 2
fontColor              = (255,255,255)
lineType               = cv2.LINE_4

# file_people = './CLASP-DATA-102319/cam09exp2_logs_seg.txt'
file_people = 'CLASP-DATA-102419/cam09exp2_logs_fullv1.txt'
file_bins = "./CLASP-DATA-102419/formatted_bins_09.txt"
file_hands = 'data/hands_noid_cam09exp2_102419.pkl'
# file_bins = "./CLASP-DATA-102319/new_bins09.txt"

# frame_start = 2407 
# frame_end   = 5500

images_dir = "./CLASP-DATA-102319/cam09exp2/"
output_dir = "viz/association_cam09exp2/"

travel_unit = {};

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)


def get_intersections(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(np.int32(boxA[0]), np.int32(boxB[0]))
    yA = max(np.int32(boxA[1]), np.int32(boxB[1]))
    xB = min(np.int32(boxA[2]), np.int32(boxB[2]))
    yB = min(np.int32(boxA[3]), np.int32(boxB[3]))
    return xA, yA, xB, yB


def compute_area(xA, yA, xB, yB):
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return interArea


def check_if_inside(xA, yA, xB, yB, hand):
    is_inside = False
    if(xA < hand[0] and xB > hand[0]):
        if(yA < hand[1] and yB > hand[1]):
            is_inside = True
    return is_inside

def restructure_bbox(box):
    # box is (x0, y0, w, h), we have to convert it to (x0, y0, x1 = x0 + w, y1 = y0 + h)
    # return (int(box[0]), int(box[1]), int(box[2]+box[0]), int(box[3]+box[1]))
    return (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

def load_travel_units():
	f = open("./travel_units.txt", 'r')
	data = f.readlines()

	for line in data:
		l = line.split(",")
		print(l)
		if len(l) == 1:
			print("No family unit!!!")
		else:
			try:
				travel_unit[l[1].replace("\n",'')].append(l[0])
			except:
				travel_unit[l[1].replace("\n",'')] = []
				travel_unit[l[1].replace("\n",'')].append(l[0])
			print("Family unit!")
	return

###### LOAD TRAVEL UNITS
load_travel_units()
print(travel_unit)

###### LOAD PEOPLE'S DATA
# 
dicti_people = {}
with open(file_people,'r') as fii:
    lines = fii.readlines()
    for line in lines:
        splitted = line.split(',')
        frame_num = int(splitted[0])
        
        if('P' in splitted[1]):
            person_id = splitted[1][1:]
        elif('T' in splitted[1]):
            person_id = splitted[1]
            
        x1 = int(float(splitted[2]))
        y1 = int(float(splitted[3]))
        x2 = int(float(splitted[4]))
        y2 = int(float(splitted[5]))
        if(frame_num in dicti_people.keys()):
            dicti_people[frame_num].append([(x1,y1,x2,y2),person_id])
        else:
            dicti_people[frame_num] = []
            dicti_people[frame_num].append([(x1,y1,x2,y2), person_id])


####### LOAD HANDS DATA
data_hands = load_obj(file_hands)
dicti_hands = {}
for i, hands in enumerate(data_hands['hands']):
    dicti_hands[i+1] = hands

####### LOAD data bins
# file_path = './binsbinsbins_09.csv'
with open(file_bins, 'r') as f:
    # frame, id, x1,y1,x2,y2
    lines = f.readlines()
    dicti_bins = {}
    for line in lines:
        splitted = line.split(',')
        frame_num = int(splitted[0])
        bin_id = int(splitted[1])
        x1 = int(splitted[2])
        y1 = int(splitted[3])
        x2 = int(splitted[4])
        y2 = int(splitted[5])
        if(frame_num in dicti_bins.keys()):
            dicti_bins[frame_num].append([(x1,y1,x2,y2),bin_id])
        else:
            dicti_bins[frame_num] = []
            dicti_bins[frame_num].append([(x1,y1,x2,y2), bin_id])


img_list = os.listdir(images_dir)
img_list.sort()

number_of_lines_on_screen = 1

bin_theft_manager = {}
imlist = []
# with open('./cam_09_exp2_associated_events.csv', 'w', newline='') as csvfile:
with open('./cam_09_exp2_associated_events.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    
    # As supposed, whole video
    for file_frame in img_list:
        frame = int(file_frame.split('.')[0].split('frame')[1])
    # for frame in dicti_people.keys():
        THEFT = False
        sys.stdout.write("\r"+str(frame))
        sys.stdout.flush()

        if frame not in dicti_bins.keys() and frame not in dicti_people.keys():
            # print('(No detection)')
            img_name =  os.path.join(images_dir, "frame"+str(frame).zfill(5)+".jpg")
            img = cv2.imread(img_name)
            imS = cv2.resize(img, (640,360))
            imlist.append((output_dir+"frame"+str(frame).zfill(5)+".jpg",imS))
            
            events = []
            spamwriter.writerow([str(frame),events])
        
        else:
            if(frame not in dicti_people.keys()):
                dicti_people[frame] = []
            if(frame not in dicti_hands.keys()):
                dicti_hands[frame] = []
            if(frame not in dicti_bins.keys()):
                dicti_bins[frame] = []

            events = []
            img_name =  os.path.join(images_dir, "frame"+str(frame).zfill(5)+".jpg")
            img = cv2.imread(img_name)
            people = dicti_people[frame]
            bins = dicti_bins[frame]
            people_hands = dicti_hands[frame]
            roi = np.zeros(img.shape, dtype=np.uint8)

            for person in people:
                # Read person information and draw it
                person_box = restructure_bbox(person[0])
                img = cv2.rectangle(img,(int(person_box[0]),int(person_box[1])),(int(person_box[2]),int(person_box[3])),(255,0,0),2)
                person_id = person[1]
                img = cv2.putText(img,str(person_id), ((int(person_box[0]),int(person_box[1]))),font,fontScale,(255,0,0),lineType)
                # Dont care about officers
                if 'TSO' in person[1]:
                    continue

                for bin in bins:
                    bin_box = bin[0]
                    img = cv2.rectangle(img,(int(bin_box[0]),int(bin_box[1])),(int(bin_box[2]),int(bin_box[3])),(0,255,0),2)
                    bin_id = bin[1]
                    img = cv2.putText(img,str(bin_id), ((int(bin_box[0]),int(bin_box[1]))),font,fontScale,(0,255,0),lineType)
                    
                    # Check if bboxes intersect
                    xA, yA, xB, yB = get_intersections(person_box, bin_box)
                    interArea = compute_area(xA, yA, xB, yB)
                    if(interArea > 0.0):
                        # The bounding boxes intersect

                        for hands in people_hands:
                            for hand in hands:
                                img = cv2.circle(img, (int(hand[0]),int(hand[1])), 10, (0,125,255), cv2.FILLED)
                                if(check_if_inside(xA, yA, xB, yB, hand)):
                                    # There is a hand inside the intersection
                                    #                                 
                                    # PASSENGER
                                    # print("PERSON " + str(person_id) + " is touching bin " + str(bin_id))
                                    if((str("P")+str(person_id) + '-' + str("B") + str(bin_id)) not in events):  
                                        # Count how many times that bin has been touched by that person
                                        if(int(bin_id) not in bin_theft_manager.keys()):
                                            # If the bin was not registered create an entry in the dictionary
                                            bin_theft_manager[int(bin_id)] = {}
                                            bin_theft_manager[int(bin_id)][person_id] = 1
                                            cv2.putText(img,"PERSON " + str(person_id) + " LINKED TO BIN " + str(bin_id),(900,50*number_of_lines_on_screen),font,fontScale,(255,255,255),lineType)
                                            number_of_lines_on_screen += 1
                                            # make the association stored in the csv file
                                            events.append(str("P")+str(person_id) + '-' + str("B") + str(bin_id))
                                        elif(person_id not in bin_theft_manager[int(bin_id)].keys()):
                                            # If another person touched the bin register it
                                            bin_theft_manager[int(bin_id)][person_id] = 1
                                        else:
                                            # A person that touched the bin touches it again, check if is a thief
                                            sorted_manager = sorted(bin_theft_manager[int(bin_id)].items(), key=operator.itemgetter(1))
                                            original_owner = list(sorted_manager)[0][0]
                                            for entry in travel_unit:
                                                # print("P" + str(original_owner))
                                                # print("P" + str(person_id))
                                                # print(entry)
                                                if "P" + str(original_owner) in travel_unit[entry] and "P" + str(person_id) in travel_unit[entry]:
                                                    print("Family unit!!! NEver mibd!!@")
                                                    # print("P" + str(original_owner))
                                                    # print("P" + str(person_id))
                                                    # print(entry)
                                            bin_theft_manager[int(bin_id)][person_id] += 1

                                            # print("Possible theif: ", person_id, " for bin ", bin_id)

                                            # input()

                                            if(len(bin_theft_manager[int(bin_id)].keys()) > 1):
                                                # Somebody else has touched the bin
                                                # Now check if there is a casual touch (don't create event), a theft (create event and freeze screen), or a shared bin (create association)
                                                if(bin_theft_manager[int(bin_id)][person_id] < 10):
                                                    # Casual touch, ignore it
                                                    print()
                                                elif(bin_theft_manager[int(bin_id)][person_id] < 20):
                                                    # Possible thief (We detected a theft but we say possible because we don't have a 100% of accuracy of detecting thiefs)
                                                    # # Security as supposed
                                                    # if(person_id == '4'):
                                                    #     continue
                                                    # else:
                                                    #     THEFT = True
                                                    #     thiefs_index = np.argmax(list(bin_theft_manager[int(bin_id)].values()))
                                                    #     thief_person_id = list(bin_theft_manager[int(bin_id)])[thiefs_index+1]
                                                    #     print("thief_person_id = " + str(thief_person_id))
                                                    #     thief_bbox = None
                                                    #     for parson in people:
                                                    #         if(parson[1]==str(thief_person_id)):
                                                    #             thief_bbox = parson[0]
                                                    #             events.append(str("P ")+str(thief_person_id) + " is possibly stoling sth from " + '-' + str("B") + str(bin_id)) 
                                                    #             img = cv2.putText(img,str(thief_person_id), ((int(thief_bbox[0]),int(thief_bbox[1]))),font,fontScale,(0,0,255),lineType)
                                                    #             img = cv2.rectangle(img,(int(thief_bbox[0]),int(thief_bbox[1])),(int(thief_bbox[2]),int(thief_bbox[3])),(0,0,255),5)
                                                    #             print(str("P")+str(thief_person_id) + " is possibly stealing sth from " + str("B") + str(bin_id))
                                                    #             cv2.putText(img,"Interesting event: " +str("P ")+str(thief_person_id) + " - " + str("B") + str(bin_id),(900,50*number_of_lines_on_screen),font,fontScale,(0,0,255),lineType)
                                                    # Security as supposed
                                                    THEFT = True
                                                    # key_list = list(bin_theft_manager[int(bin_id)].keys())
                                                    # val_list = list(bin_theft_manager[int(bin_id)].values())
                                                    # thiefs_index = np.argmax(val_list)
                                                    # print(bin_theft_manager[int(bin_id)])
                                                    

                                                    sorted_manager = sorted(bin_theft_manager[int(bin_id)].items(), key=operator.itemgetter(1))
                                                    thief_person_id = list(sorted_manager)[1][0]
                                                    original_owner = list(sorted_manager)[0][0]
                                                    print("thief_person_id = " + str(thief_person_id))
                                                    print("original_owner = " + str(original_owner))
                                                    thief_bbox = None
                                                    for parson in people:
                                                        if(parson[1]==str(thief_person_id)):
                                                            thief_bbox = parson[0]
                                                            events.append(str("P ")+str(thief_person_id) + " is possibly stoling sth from " + '-' + str("B") + str(bin_id)) 
                                                            img = cv2.putText(img,str(thief_person_id), ((int(thief_bbox[0]),int(thief_bbox[1]))),font,fontScale,(0,0,255),lineType)
                                                            img = cv2.rectangle(img,(int(thief_bbox[0]),int(thief_bbox[1])),(int(thief_bbox[2]),int(thief_bbox[3])),(0,0,255),5)
                                                            print(str("P")+str(thief_person_id) + " is possibly stealing sth from " + str("B") + str(bin_id))
                                                            cv2.putText(img,"Interesting event: " +str("P ")+str(thief_person_id) + " - " + str("B") + str(bin_id),(900,50*number_of_lines_on_screen),font,fontScale,(0,0,255),lineType)
                                                            number_of_lines_on_screen += 1
                                    roi = cv2.rectangle(roi,(int(xA),int(yA)),(int(xB),int(yB)),(0,255,0),cv2.FILLED)
            spamwriter.writerow([str(frame),events])
            events = []
            number_of_lines_on_screen = 1
            n_img = cv2.addWeighted(img, 0.7, roi, 0.3, 0)
            imS = cv2.resize(n_img, (640,360))
            
            if(THEFT):
                # Stop the scene when there is a theft
                for i in range(0,15):
                    imlist.append((os.path.join(output_dir, "frame"+str(frame).zfill(5)+".jpg"),imS))
                    # cv2.imwrite(os.path.join(output_dir, lista[frame+1]),imS)
            else:
                # cv2.imwrite(os.path.join(output_dir, lista[frame+1]),imS)
                imlist.append((os.path.join(output_dir, "frame"+str(frame).zfill(5)+".jpg"),imS))
            # cv2.imshow('im', imS)
            # cv2.waitKey(15)

# This line below should be run to save an object to later give it to camera 11 associations
save_obj(bin_theft_manager,'./theft_manager_dict.pkl')
print("WRITING IMGS")
for i,im in enumerate(imlist):
    # cv2.imwrite(os.path.join(output_dir,str(i+frame_start)+".jpg"),im[1])
    cv2.imwrite(os.path.join(output_dir,str(i).zfill(5)+".jpg"),im[1])


# ffmpeg -r 30 -f image2 -start_number 4202 -i %07d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p cam09exp2_pink_results.mp4
