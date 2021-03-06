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
from skimage.transform import resize
import random
import sys
import argparse
from tvnet1130_train_options import arguments
import ast
import time
import pickle

def actions(act):
    if act == "46":
        return "push"
    elif act == "36":
        return "pick up"

def visualizeFlow(f):
    flow = np.zeros((240, 320, 2))
    flow[:, :, 0] = f[0,:,:]# .cpu().numpy()
    flow[:, :, 1] = f[1,:,:]# .cpu().numpy()
    hsv = np.zeros((240, 320, 3), dtype=np.uint8)

    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = bgr[...,::-1] # comment out when saving with cv2.imwrite

    return rgb


class ava_dataset(data.Dataset):
    def __init__(self, folderList, frameDir, flowDir, jsonDir):
        self.folderList = folderList;
        self.frameDir = frameDir;
        self.jsonDir = jsonDir;
        self.flowDir = flowDir;

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
        
        return height, width

    def within(self, keypoints, tup):
        score = 0
        new_keypoints = np.zeros((17,3))

        i = 0
        for item in keypoints:
            if tup[0] <= item[0] <= tup[1] and  tup[2] <= item[1] <= tup[3]:
                new_keypoints[i, :] = [item[0], item[1], item[2]]
                score = score + 1
            i = i + 1

        return score, new_keypoints

    # (altered but thank you): https://github.com/leaderj1001/Action-Localization/tree/master/video_crawler 
    def prepare(self, sample, video_path, jsonDir):
        tube = {}
        video_name = sample[0]
        start_time = int(sample[1])
        start_time = int(start_time) - 1
        end_time = int(start_time) + 3
        
        print("\n" + str(sample))

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
            pass # print('Video does not exist: {0}'.format(video_name))
        elif os.path.isfile(cropped_video_filename):
            pass # print('Already exist cropped video: {0}'.format(video_name))
        else:
            command = ['ffmpeg','-i','"%s"' % origin_video_filename, '-ss', str(start_time), '-t', str(end_time - start_time), '-c:v', 'libx264', '-c:a', 'ac3','-threads', '1','-loglevel', 'panic','"{}"'.format(cropped_video_filename)]
            command = ' '.join(command)

            try:
                # print("Processing video: {}".format(video_name))
                output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                # print('status :: ', status, ', error print :: ', err.output.decode('euc-kr'))
                return status, err.output
            
            os.mkdir(cropped_video_filename.split(".")[0])
            os.mkdir(cropped_video_filename.split(".")[0] + "/scene/")
            os.mkdir(cropped_video_filename.split(".")[0] + "/rgb/")
            os.mkdir(cropped_video_filename.split(".")[0] + "/joints/")
            os.mkdir(cropped_video_filename.split(".")[0] + "/flow/")
            command = "ffmpeg -i " + cropped_video_filename + " -vf \"select=not(mod(n\,1))\" -vsync vfr -q:v 2 "  + cropped_video_filename.split(".")[0] + "/scene/%07d.jpg"
            # print(command)
            
            
            try:
                # print("Splicing video: {}".format(video_name))
                output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                # print('status :: ', status, ', error print :: ', err.output.decode('euc-kr'))
                return status, err.output

            command = "cd ./detectron2 && /home/truppr/anaconda3/envs/detectron2/bin/python demo/run.py --config-file configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --video-input " + str(cropped_video_filename) + " --output " +  cropped_video_filename.split(".")[0] + "/joints/" + " --opts MODEL.WEIGHTS ~/model_final_997cc7.pkl"
            # print(command)
            
            try:
                # print("Capturing Pose INFO: {}".format(video_name))
                output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                # print('status :: ', status, ', error print :: ', err.output.decode('euc-kr'))
                return status, err.outputs

            command = "cd ./OF && /home/truppr/anaconda3/envs/py36pt110/bin/python main.py --optical_weights flownet2-pytorch/models/FlowNet2_checkpoint.pth.tar --image_dir " + cropped_video_filename.split(".")[0] + "/scene/" + " --save_dir " + cropped_video_filename.split(".")[0] + "/flow/"
            print(command)
            try:
                output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                return status, err.outputs

            path = cropped_video_filename.split(".")[0]
            flow_data = {}
            joint_data = np.zeros((len(os.listdir(path + "/scene/")), 17, 3))
            joint_index = 0

            # NOTE: it is necessary to move this to after the joint processing, if you would like to use the Pose Coordinates to draw BB as opposed to using GT BB.
            for filename in sorted(os.listdir(path + "/scene/")):
                if filename.endswith(".jpg"): 
                    i = imread(path + "/scene/" + filename)
                    flow_data[path + "/scene/" + filename] = []
                    

                    buffer = 25
                    ymin = int(float(sample[3]) * height) - buffer
                    ymax = int(float(sample[5]) * height) + buffer
                    xmin = int(float(sample[2]) * width) - buffer
                    xmax = int(float(sample[4]) * width) + buffer

                    if xmin < 1:
                        xmin = 1
                    if ymin < 1:
                        ymin = 1
                    if xmax >= width:
                        xmax = width - 1 
                    if ymax >= height:
                        ymax = height - 1 

                    flow_data[path + "/scene/" + filename] = [xmin, ymin, xmax, ymax]

                    # Uncomment to see where the RGB stream is taken from within the scene...
                    # scene = cv2.rectangle(i, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                    # scene = cv2.rectangle(scene, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                    # scene = cv2.putText(scene, "Person " + sample[7], (xmin + 10, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
                    # scene = cv2.putText(scene, "Action " + sample[6], (xmin + 10, ymin + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
                    # imsave(path + "/scene/" + filename, scene)
                    
                    original_image = np.transpose(i, (2, 0, 1))
                    i = i[ymin:ymax,xmin:xmax,:]
                    # i = skimage.transform.resize(i, (224,224, 3))
                    i = resize(i, (224,224, 3))
                    
                    imsave(path + "/rgb/" + filename, i)

                else:
                    continue
            fuck_up = 0;
            
            for filename in sorted(os.listdir(cropped_video_filename.split(".")[0] + "/joints/")):
                if filename.endswith(".txt") and "-jo-" in filename.lower():

                    f = open(cropped_video_filename.split(".")[0] + "/joints/" + filename, "r")


                    frame = filename.split("-")[-1].replace(".txt", "")

                    try:
                        joint_data[joint_index, :, :]
                    except Exception as e:
                        break

                    s = f.readlines()[0].strip()

                    if s[0:2] != "[]":
                        joints = np.array(eval(s))
                        min_score = 0
                        best = -1

                        for x in range(0, joints.shape[0]):
                            score, joints[x, :, :] = self.within(joints[x, :, :], [xmin, xmax, ymin, ymax])
                            if score > min_score:
                                min_score = score
                                best = x
                        
                        if best != -1:
                            joint_data[joint_index, :, :] = joints[best, :, :]

                        else:
                            joint_data[joint_index, :, :] = np.zeros((17,3))
                    else:
                        joint_data[joint_index, :, :] = np.zeros((17,3))
                        fuck_up = fuck_up + 1;

                    f.close()
                    joint_index = joint_index + 1

            # print("Reportoing fuckups: ", fuck_up, " from ", joint_index, " joints.")

            ehpi = np.zeros((17, joint_data.shape[0], 3))
            for k in range(0, joint_data.shape[0]):
                # print(joint_data[k, :, 0] , " - ", xmin, "/", xmax, "-", xmin)
                ehpi[:, k, 0] = ((joint_data[k, :, 0] - xmin) / (xmax - xmin)) * 255
                ehpi[:, k, 1] = ((joint_data[k, :, 1] - ymin) / (ymax - ymin)) * 255
                ehpi[:, k, 2] = 0
            
            ehpi = ehpi.clip(0, 255).astype(int)
            np.save(cropped_video_filename.split(".")[0] + "/joints/ehpi.npy", ehpi)
            imsave(cropped_video_filename.split(".")[0] + "/joints/ehpi.jpg", ehpi)

            with open(cropped_video_filename.split(".")[0] + "/scene/" + 'results.json', 'w') as fp:
                json.dump(str(flow_data).replace("array","np.array").strip(), fp)

        with open(cropped_video_filename.split(".")[0] + "/scene/" + 'results.json', 'r') as json_file:
            flow_data = json.load(json_file)

        flow_data = eval(flow_data)

        # joint_data = eval(joint_data)
        joint_data = np.load(cropped_video_filename.split(".")[0] + "/joints/ehpi.npy")
        # print(joint_data)

        return cropped_video_filename.split(".")[0], flow_data, joint_data, height, width

    def extractFlow(self, path, tube, h, w, bb):
        flow_frame = torch.zeros((1, 2, len(bb.keys()), 224, 224)) # (1, 2, 64, 224, 224)

        index = 0;
        for entry in sorted(bb.keys()):
            frame = str(int(entry.split("/")[-1].replace(".jpg", ""))) + ".pkl"
            # print(path + "/flow/", frame)

            try:
                with open(path + "/flow/" + frame, 'rb') as f:
                    data = pickle.load(f)
            except FileNotFoundError:
                if index != 0:
                    flow_frame[0, :, index, :, :] = flow_frame[0, :, index - 1, :, :]
                continue

            f = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
            f = data
            flow_frame[0, :, index, :, :] = torch.from_numpy(np.resize(f[:,bb[entry][0]:bb[entry][2], bb[entry][1]:bb[entry][3]], (2, 224,224)))

            index = index + 1

        return flow_frame


    def extractRGB(self, path, tube, h, w, flow_data):
        rgb_frame = torch.zeros((1, 3, len(flow_data.keys()), 224, 224))

        index = 0
        for filename in sorted(os.listdir(path + "/rgb/")):
            try:
                i = imread(path + "/rgb/" + filename)
            except FileNotFoundError:
                if index != 0:
                    rgb_frame[:, :, index, :, :] = rgb_frame[:, :, index - 1, :, :]
                continue

            i = np.transpose(i, (2, 0, 1))
            i = resize(i, (3, 224, 224))
            rgb_frame[:, :, index, :, :] = torch.from_numpy(i)
            index = index + 1

        return rgb_frame

    def __len__(self):
        with open(self.folderList, 'r') as file:
            sample = list(csv.reader(file))
        return len(sample)


    def __getitem__(self, index):
        tube = {'rgb' : torch.zeros((1, 3, 64, 224, 224)), 'of': torch.zeros((1, 2, 64, 224, 224)), 'joints' : torch.zeros((17,64,3)), "action" : -1}

        # Step 1 - iterate over training list
        with open(self.folderList, 'r') as file:
            sample = list(csv.reader(file))[index]
        
        # try:
        # Step 2 - check if preprocessed data for sample exists
        # now = time.time()
        try:
            path, flow_data, joint_data, h, w = self.prepare(sample, self.frameDir, self.jsonDir)
        except:
            return tube
        # print(time.time() - now)

        length = len(flow_data.keys())
        startIndex = random.randint(0,length - 64)

        
        flow = self.extractFlow(path, tube, h, w, flow_data)
        rgb = self.extractRGB(path, tube, h, w, flow_data)

        tube["joints"] = torch.from_numpy(joint_data[:, startIndex:startIndex + 64, :])
        tube["rgb"] = rgb[:, :, startIndex:startIndex + 64, :, :]
        tube["of"] = flow[:, :, startIndex:startIndex + 64, :, :]
        tube["action"] = actions(sample[6])

        '''
        # Visualize Tubes
        for frame in range(0, tube["scene"].shape[2]):
            # im = Image.fromarray(tube["scene"][0,:,frame,:,:].numpy(), "RGB")
            im = np.transpose(tube["scene"][0,:,frame,:,:].numpy(), (1, 2, 0))
            print(im.shape)
            imsave(path + "/scene/test-"+ str(frame).zfill(7) + ".tiff", im)
            # im = Image.fromarray(im, "RGB")
            # im.save(path + "/scene/test-"+ str(frame).zfill(7) + ".jpg")

            # im = Image.fromarray(tube["rgb"][0,:,frame,:,:].numpy(), "RGB")
            im = np.transpose(tube["rgb"][0,:,frame,:,:].numpy(), (1, 2, 0))
            print(im.shape)
            imsave(path + "/rgb/test-"+ str(frame).zfill(7) + ".tiff", im)
            # im = Image.fromarray(im, "RGB")
            # im.save(path + "/rgb/test-"+ str(frame).zfill(7) + ".jpg")
        '''

        return tube