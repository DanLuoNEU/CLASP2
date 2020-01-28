# Using hands ID to associate the bin with persons and detect possible thefts
# At every frame, bins are the main parts, it will have all the owners in the record file
# 1. Decide bins' belonging
# 2. Detect if there is any suspicious activity
# Dan, 12/15/2019
########## Import ##########
import os
import sys
import cv2
import csv
import time
import pickle
import operator
import numpy as numpy
import scipy.io as sio
import matplotlib.pyplot as plt
########## Configuration ##########
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (900,50)
fontScale              = 2
fontColor              = (255,255,255)
lineType               = cv2.LINE_4
# Files to load
# Data forms
## bins
### .csv file: frame, id, x1, y1, x2, y2
## handsID:
### persons: dictionary{
#            - keys: 'bbox', 'bins', 'hands', 'id'
#            - values: dictionary{
#                      - keys: frame
#                      - values: list of data
#                      }
# }
THD_OWNER = 30
cam = str(11).zfill(2)
file_bins = "CLASP-DATA-102419/training/formatted_bins_{}.txt".format(cam)
file_handsID = 'data/hands_id_cam{}exp2_102419.pkl'.format(cam)
file_assoc = 'events/events_cam{}exp2_102419.csv'.format(cam)
file_manager = 'data/bin_manager.pkl'.format(cam)
file_tu = 'travel_units.txt'
# Directories used later 
images_dir = "./CLASP-DATA-102319/cam{}exp2/".format(cam)
output_dir = "viz/association_handsID_cam{}exp2/".format(cam)

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

def check_if_inside(bin, hand):
    xA, yA, xB, yB = bin[0][0], bin[0][1], bin[0][2], bin[0][3]
    is_inside = False
    if(xA < hand[0] and xB > hand[0]):
        if(yA < hand[1] and yB > hand[1]):
            is_inside = True
    return is_inside

###### LOAD PEOPLE'S DATA
def load_dicti(file_handsID=file_handsID, file_bins=file_bins):
    ####### LOAD People,Hands ID DATA
    dicti_persons = load_obj(file_handsID)

    ####### LOAD data bins
    with open(file_bins, 'r') as f:
        # frame, id, x1, y1, x2, y2
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
    
    ####### LOAD Traveling Units
    with open(file_tu, 'r') as f:
        # Pid, (TU if there is one)
        lines = f.readlines()
        dicti_tu = {}
        for line in lines:
            splitted = line.split(',')
            if len(splitted) == 1:
                tu = splitted[0].split('\n')[0]
                dicti_tu[tu] = tu
            else:
                P = splitted[0].split('\n')[0]
                tu = splitted[1].split('\n')[0].replace(' ','')
                dicti_tu[P] = tu

    return dicti_persons, dicti_bins,dicti_tu

def main():
    # Make the directory used to store pictures
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Prepare the loaded data
    dicti_hands, dicti_bins, dicti_tu = load_dicti()
    img_list = os.listdir(images_dir)
    img_list.sort()
    # Initialize the event manager
    if cam == '09':
        bin_manager = {}
    else:
        bin_manager = load_obj(file_manager)
    imlist = [] # for visualization
    with open(file_assoc,'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')

        for file_frame in img_list:
            events = []
            POSSIBLE_THEFT = False

            frame = int(file_frame.split('.')[0].split('frame')[1])
            sys.stdout.write("\r"+str(frame))
            sys.stdout.flush()

            # Use the bin as the body of the event
            ## when there is no detected bins
            if frame not in dicti_bins.keys():
                img_name =  os.path.join(images_dir, "frame"+str(frame).zfill(5)+".jpg")
                img = cv2.imread(img_name)
                # TODO: people visualization here
                imS = cv2.resize(img, (640,360))
                imlist.append((output_dir+"frame"+str(frame).zfill(5)+".jpg",imS))
                if frame in dicti_hands['id'].keys():
                    pid_str = ''
                    for pid in dicti_hands['id'][frame]:
                        pid_str = pid_str + pid + ' '
                    # events.append(pid_str + 'detected')
                else:
                    # events = [" No detected bins and people! "]
                    events = []
            else:
                # Avoid there is no hand detection information
                if frame not in dicti_hands['hands'].keys():
                    dicti_hands['hands'][frame] = []
                
                events = []
                for bin in dicti_bins[frame]:
                    # Register in bin_manager when the bin first shows
                    if bin[1] not in bin_manager.keys():
                        bin_manager[bin[1]]={}
                    # If there are hands detected and they are not zero
                    hands_list = dicti_hands['hands'][frame]
                    if hands_list != []:
                        for i, hands in enumerate(hands_list):
                            if hands != []:
                                for j, hand in enumerate(hands):
                                    pid = dicti_hands['id'][frame][i]
                                    if hand[0]!=0 or hand[1]!=0:
                                        if check_if_inside(bin,hand):
                                            if j==0:
                                                events.append('|'+pid+'| right hand in |Bin '+str(bin[1])+'|')
                                            elif j==1:
                                                events.append('|'+pid+'| left hand in |Bin '+str(bin[1])+'|')
                                            # End of condition right hand or left hand
                                            if 'owner' not in bin_manager[bin[1]].keys():
                                                if pid not in bin_manager[bin[1]].keys():
                                                    bin_manager[bin[1]][pid] = 1
                                                elif bin_manager[bin[1]][pid] < THD_OWNER:
                                                    # The first count up to the threshold is the owner of the bin
                                                    bin_manager[bin[1]][pid] = bin_manager[bin[1]][pid] + 1
                                                else:
                                                    bin_manager[bin[1]]['owner'] = pid
                                                    events.append('|'+pid+'| is the owner of |Bin '+str(bin[1])+'|')
                                                # End of condition person id setup in bin_manager for bin[1] or not
                                            else:
                                                # events.append('|'+bin_manager[bin[1]]['owner']+'| is the owner of |Bin '+str(bin[1])+'|')
                                                if pid != bin_manager[bin[1]]['owner']:
                                                    if (dicti_tu[pid] != dicti_tu[bin_manager[bin[1]]['owner']]) and ('TSO' not in pid):
                                                        events.append('|'+pid+'| is suspicious with |Bin '+str(bin[1])+'|')
                                                # End of condition person id is not the owner of the bin[1]
                                            # End of condition that owner exists or not 
                                        # End of condition hand is in bin's bounding box
                                    # End of condition that hand is existed
                                # End of loop of hands
                            # End of condition hands!=[]
                        # End of loop of hands_list
                    # End of condition hands_list != []
                # End of bins loop
            spamwriter.writerow([str(frame),events])
            # End of if frame in dicti_bins.keys() or not
        # End for loop for the img_list
    # End opening the .csv file
    save_obj(bin_manager, file_manager)
    print('Well Done!')
# End of main function

if __name__ == '__main__':
    main()