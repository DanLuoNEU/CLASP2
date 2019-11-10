import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import skimage
import operator
import os
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
import re
import random
from torchvision.transforms import ToTensor
import csv
from random import randrange
import subprocess
import cv2
import shlex
import json
from skimage.io import imread, imsave

def loadTVNet(model, tvnet_init):
        loadedcheckpoint = torch.load(tvnet_init, map_location=(lambda storage, loc: storage) if torch.cuda.is_available() else 'cpu')
        stateDict = loadedcheckpoint['state_dict']

        own_state = model.state_dict()
        for name, param in stateDict.items():
                if name not in own_state:
                        continue
                if isinstance(param, nn.Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                own_state[name].copy_(param)
        model.load_state_dict(own_state)

        return model

########## TVNet for Optical Flow Extraction for I3D Flow Stream
'''
from model.network import model as tvnet
tvnet_args = arguments().parse() # tvnet parameters
tvnet_args.batch_size = 4; # 3 + 1 for UCF101
tvnet_args.data_size = [tvnet_args.batch_size, 3, 240, 320]
# init_file = '/home/truppr/kDYAN-TV/savedModels/UCF-TVNet-1-1-30.pth'
init_file = '/home/truppr/kDYAN-TV/savedModels/UCF-DYAN-TVNET-1-1-30-E2E-DL_1_0.3163.pth'
of = tvnet(tvnet_args).cuda()

of = loadTVNet(of, init_file)

of = of.cpu()
'''

class ava_dataset(data.Dataset):
    def __init__(self, folderList, frameDir, flowDir, jsonDir):
        self.folderList = folderList;
        self.frameDir = frameDir;
        self.jsonDir = jsonDir;
        self.flowDir = flowDir;
        # self.numpixels = 320 * 240 # 640 * 480 # 1920*1080
        # self.videoFrameProg = 1;

    # function to find the resolution of the input video file
    # (thank you): https://gist.github.com/oldo/dc7ee7f28851922cca09
    def findVideoMetada(self, pathToInputVideo):
        cmd = "ffprobe -v quiet -print_format json -show_streams"
        args = shlex.split(cmd)
        args.append(pathToInputVideo)
        # run the ffprobe process, decode stdout into utf-8 & convert to JSON
        ffprobeOutput = subprocess.check_output(args).decode('utf-8')
        ffprobeOutput = json.loads(ffprobeOutput)

        # prints all the metadata available:
        import pprint
        pp = pprint.PrettyPrinter(indent=2)
        # pp.pprint(ffprobeOutput)

        # for example, find height and width
        height = ffprobeOutput['streams'][0]['height']
        width = ffprobeOutput['streams'][0]['width']
        
        # print(height, width)
        return height, width

    def within(self, keypoints, tup):
        boolean = False
        score = 0

        for item in keypoints:
            if tup[0] <= item[0] <= tup[1] and  tup[2] <= item[1] <= tup[3]:
                boolean = True;
                score = score + 1

        print(keypoints)
        print(tup)

        return boolean, score

    # (altered but thank you): https://github.com/leaderj1001/Action-Localization/tree/master/video_crawler 
    def prepare(self, sample, video_path, jsonDir, mode='trainval'):
        tube = {}
        video_name = sample[0]
        start_time = int(sample[1])
        start_time = int(start_time) - 1
        end_time = int(start_time) + 3
        print(sample)
        # print("Searching ", video_path, "for", video_name)
        

        onlyfiles = [f for f in listdir(video_path) if isfile(join(video_path, f))]
        for f in onlyfiles:
            if video_name in f:
                video_name = f
                origin_video_filename = '{}/{}'.format(video_path, video_name)
                extension = video_name.split(".")[1]
        cropped_video_filename = '{}/{}'.format(jsonDir, video_name.split(".")[0] + "_" + sample[1] + "_" + sample[6] + "_" + sample[7] + "." + extension)

        height, width = self.findVideoMetada(origin_video_filename)

        status = False
        if not os.path.isfile(origin_video_filename):
            print('Video does not exist: {0}'.format(video_name))
        elif os.path.isfile(cropped_video_filename):
            print('Already exist cropped video: {0}'.format(video_name))
        else:
            command = ['ffmpeg','-i','"%s"' % origin_video_filename, '-ss', str(start_time), '-t', str(end_time - start_time), '-c:v', 'libx264', '-c:a', 'ac3','-threads', '1','-loglevel', 'panic','"{}"'.format(cropped_video_filename)]
            command = ' '.join(command)
            print(command)

            try:
                print("\tProcessing video: {}".format(video_name))
                output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                print('status :: ', status, ', error print :: ', err.output.decode('euc-kr'))
                return status, err.output

            # if os.path.exists(cropped_video_filename)
            
            os.mkdir(cropped_video_filename.split(".")[0])
            os.mkdir(cropped_video_filename.split(".")[0] + "/scene/")
            os.mkdir(cropped_video_filename.split(".")[0] + "/rgb/")
            os.mkdir(cropped_video_filename.split(".")[0] + "/joint/")
            command = "ffmpeg -i " + cropped_video_filename + " -vf \"select=not(mod(n\,1))\" -vsync vfr -q:v 2 "  + cropped_video_filename.split(".")[0] + "/scene/%07d.jpg"
            print(command)

            try:
                print("Splicing video: {}".format(video_name))
                output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                print('status :: ', status, ', error print :: ', err.output.decode('euc-kr'))
                return status, err.output
            
            # Extract JOINT info
            img_directory = cropped_video_filename.split(".")[0] + "/scene/" 
            command = "cd ./AlphaPose/ && /home/truppr/anaconda3/envs/dyanEnv/bin/python demo.py --indir " + img_directory + " --outdir " + cropped_video_filename.split(".")[0] + "/joint/" + " --fast_inference False"
            print(command)

            try:
                print("Capturing Pose INFO: {}".format(video_name))
                output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                print('status :: ', status, ', error print :: ', err.output.decode('euc-kr'))
                return status, err.output

            with open(cropped_video_filename.split(".")[0] + "/joint/" + 'alphapose-results.json') as json_file:
                data = json.load(json_file)
            
            print("extracting data...")
            i = 0;
            new_data = {}
            for entry in data:
                try:
                    # print("try")
                    new_data[str(data[i]["category_id"])]
                except KeyError:
                    # print("caught")
                    new_data[str(data[i]["category_id"])] = {}
                '''
                try:
                    print("try")
                    new_data[str(data[i]["category_id"])][data[i]["image_id"]]
                except KeyError:
                    print("caught")
                    new_data[str(data[i]["category_id"])][data[i]["image_id"]] 
                '''
                # print("out")
                new_data[str(data[i]["category_id"])][data[i]["image_id"]] = []
                for eh in range(0, len(data[i]["keypoints"]), 3):
                    new_data[str(data[i]["category_id"])][data[i]["image_id"]].append((data[i]["keypoints"][eh], data[i]["keypoints"][eh + 1], data[i]["keypoints"][eh + 2]))  
                
                i = i + 1

            print("extracting relevent data...")
            friend = -1 # this is the person who we are interested in...
            friend_score = {}
            for person in new_data:
                for frame in new_data[person]:
                    res, num = self.within(new_data[person][frame], (int(float(sample[3]) * height), int(float(sample[5]) * height), int(float(sample[2]) * width), int(float(sample[4]) * width)))
                    if res:
                        try:
                            print("try")
                            friend_score[str(person)]
                        except KeyError:
                            print("caught")
                            friend_score[str(person)] = 0
                        friend = person
                        friend_score[person] = friend_score[person] + num

            if friend == -1:
                print("Uh oh...")
            else:
                print("is this new data?")
                print("person ", str(max(friend_score.iteritems(), key=operator.itemgetter(1))[0]))
                peep = str(max(friend_score.iteritems(), key=operator.itemgetter(1))[0])
                new_data = new_data[peep]

            print(new_data)

            with open(cropped_video_filename.split(".")[0] + "/joint/" + 'results.json', 'w') as fp:
                json.dump(new_data, fp)
            # input()
        '''
        rgb_stream = torch.zeros(1, 3, 90, 224, 224) # torch.zeros(1,3,64,224,224)
        onlyfiles = [f for f in listdir(cropped_video_filename.split(".")[0] + "/scene/") if isfile(join(cropped_video_filename.split(".")[0] + "/scene/", f))]
        index = 0;
        for f in sorted(onlyfiles):
            i = imread(join(cropped_video_filename.split(".")[0] + "/scene/", f))

            # Visualize bounding boxes...
            #i = cv2.rectangle(i, (int(float(sample[2]) * width), int(float(sample[3]) * height)), (int(float(sample[4]) * width), int(float(sample[5]) * height)), (255, 255, 0), 2)
            #if int(f.split(".")[0]) > 30 and int(f.split(".")[0]) < 60:
            #    i = cv2.putText(i, "Person " + sample[7], (int(float(sample[2]) * width) + 10, int(float(sample[3]) * height) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
            #    i = cv2.putText(i, "Action " + sample[6], (int(float(sample[2]) * width) + 10, int(float(sample[3]) * height) + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
            

            # Save RGB Stream Input

            i = i[int(float(sample[3]) * height):int(float(sample[5]) * height),int(float(sample[2]) * width):int(float(sample[4]) * width),:]
            i = skimage.transform.resize(i, (224,224, 3))
            imsave(join(cropped_video_filename.split(".")[0] + "/rgb/", f), i)
            i = np.transpose(i, (2, 0, 1))

            rgb_stream[0,:,index,:,:] = torch.from_numpy(i)
            index -= -1;

        '''
        # eturn rgb_stream
        

        return new_data

    def viableActionTube(self, imgs):
        previous = 0
        count = 0;
        for im in imgs:
            # print(im)
            # print(int(im.replace(".jpg",'')) - previous)

            if int(im.replace(".jpg",'')) - previous < 2:
                count = count + 1;
            else:
                if count == 0:
                    pass
                else:
                    input("Interruption in tube!!!")
                count = 0;
                # continue
            previous = int(im.replace(".jpg",''))
            # print()

        if count >= 64:
            return True
        else:
            return False

    def extractTube(self):
        return

    def DataAugmentation(self, tube):
        return

    def __len__(self):
        return len(self.folderList)


    def __getitem__(self, index):
        tube = {'rgb' : torch.zeros((224,224,3)), 'of': torch.zeros((224,224,2)), 'joints' : []}

        # Step 1 - iterate over training list
        with open(self.folderList, 'r') as file:
            sample = list(csv.reader(file))[index]

        # Step 2 - check if preprocessed data for sample exists
        joint_info = self.prepare(sample, self.frameDir, self.jsonDir)
        
        
        # Step 3 - Extract Tube
        # tube = self.extractTube(tube)

        '''
        # Step 4 - Determine Augmentation
        if not self.viableActionTube(tube):
            tube = self.DataAugmentation(tube)
        elif(randrange(10) > 3):
            tube = self.DataAugmentation(tube)

        # step 5 - return tube
        '''
        return tube