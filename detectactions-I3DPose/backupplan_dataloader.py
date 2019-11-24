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
import random
import sys
import argparse
from tvnet1130_train_options import arguments
import ast
import time

'''
import detectron2.detectron2.utils.comm as comm
from detectron2.detectron2.checkpoint import DetectionCheckpointer
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.detectron2.utils.logger import setup_logger
'''

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


def frame_to_tensor(frame):
	# img_path1 = os.path.join(path, str('%07d' % framenum) + '.jpg')       
	image1 = Image.open(frame)
	image1 = tf1(image1)
	image1 = ToTensor()(image1)
  
	return image1


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
sys.path.insert(0, './model')
from network import model as tvnet
tvnet_args = arguments().parse() # tvnet parameters
tvnet_args.batch_size = 4; # 3 + 1 for UCF101
tvnet_args.data_size = [tvnet_args.batch_size, 3, 240, 320]
# init_file = '/home/truppr/kDYAN-TV/savedModels/UCF-TVNet-1-1-30.pth'
init_file = '/home/truppr/kDYAN-TV/savedModels/UCF-DYAN-TVNET-1-1-30-E2E-DL_1_0.3163.pth'
of = tvnet(tvnet_args).cuda()

of = loadTVNet(of, init_file)

# of = of.cpu()
'''

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
        boolean = False
        score = 0

        for item in keypoints:
            if tup[0] <= item[0] <= tup[1] and  tup[2] <= item[1] <= tup[3]:
                boolean = True;
                score = score + 1

        return boolean, score

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
            print('Video does not exist: {0}'.format(video_name))
        elif os.path.isfile(cropped_video_filename):
            print('Already exist cropped video: {0}'.format(video_name))
        else:
            command = ['ffmpeg','-i','"%s"' % origin_video_filename, '-ss', str(start_time), '-t', str(end_time - start_time), '-c:v', 'libx264', '-c:a', 'ac3','-threads', '1','-loglevel', 'panic','"{}"'.format(cropped_video_filename)]
            command = ' '.join(command)

            try:
                print("Processing video: {}".format(video_name))
                output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                print('status :: ', status, ', error print :: ', err.output.decode('euc-kr'))
                return status, err.output
            
            os.mkdir(cropped_video_filename.split(".")[0])
            os.mkdir(cropped_video_filename.split(".")[0] + "/scene/")
            os.mkdir(cropped_video_filename.split(".")[0] + "/rgb/")
            # os.mkdir(cropped_video_filename.split(".")[0] + "/joints/")
            os.mkdir(cropped_video_filename.split(".")[0] + "/flow/")
            command = "ffmpeg -i " + cropped_video_filename + " -vf \"select=not(mod(n\,1))\" -vsync vfr -q:v 2 "  + cropped_video_filename.split(".")[0] + "/scene/%07d.jpg"
            # print(command)
            
            
            try:
                print("Splicing video: {}".format(video_name))
                output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                print('status :: ', status, ', error print :: ', err.output.decode('euc-kr'))
                return status, err.output
            
            path = cropped_video_filename.split(".")[0]
            for filename in sorted(os.listdir(path + "/scene/")):
                if filename.endswith(".jpg"): 
                    #   print(filename)
                    i = imread(path + "/scene/" + filename)

                    ymin = int(float(sample[3]) * height)
                    ymax = int(float(sample[5]) * height)
                    xmin = int(float(sample[2]) * width)
                    xmax = int(float(sample[4]) * height)

                    buffer = 25
                    # print(x_arr)
                    # input()
                    xmin = int(min(x_arr)) - buffer 
                    xmax = int(max(x_arr)) + buffer 
                    ymin = int(min(y_arr)) - buffer
                    ymax = int(max(y_arr)) + buffer

                    if xmin < 0:
                        xmin = 1
                    if ymin < 0:
                        ymin = 1
                    if xmax > width:
                        xmax = width - 1 
                    if ymax > height:
                        ymax = height - 1 

                    scene = cv2.rectangle(i, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                    scene = cv2.rectangle(scene, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                    scene = cv2.putText(scene, "Person " + sample[7], (xmin + 10, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
                    scene = cv2.putText(scene, "Action " + sample[6], (xmin + 10, ymin + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

                    original_image = np.transpose(i, (2, 0, 1))
                    i = i[ymin:ymax,xmin:xmax,:]
                    i = skimage.transform.resize(i, (224,224, 3))
                    # imsave(join(path + "/rgb/", item + ".jpg"), i)

                    imsave(path + "/scene/" + filename, scene)
                    imsave(path + "/rgb/" + filename, i)

                else:
                    continue
            
        return cropped_video_filename.split(".")[0], height, width

    def extractFlow(self, path, tube, h, w, bb):
        flow_frame = torch.zeros((64, 2, 240, 320))
        path = path + "/flows/"
        os.mkdir(path)

        print(tube.shape)
        for index in range(1, 64):
            data = skimage.transform.resize(tube[0,:,index - 1, :, :].cpu().numpy(), (3, 240, 320))
            x1 = torch.from_numpy(data).cuda()
            data = skimage.transform.resize(tube[0,:,index, :, :].cpu().numpy(), (3, 240, 320))
            x2 = torch.from_numpy(data).cuda()
            print(x2.type())
            u1, u2, _ = of(x2.unsqueeze(0), x1.unsqueeze(0), need_result=True)

            data = u1.detach()[0, 0, bb[index][1]:bb[index][3], bb[index][0]:bb[index][2]]
            data = skimage.transform.resize(data.cpu().numpy(), (1, 1, 224, 224))
            flow_frame[index, 0,:,:] = u1 # torch.from_numpy(data)

            data = u2.detach()[0, 0, bb[index][1]:bb[index][3], bb[index][0]:bb[index][2]]
            data = skimage.transform.resize(data.cpu().numpy(), (1,1, 224, 224))
            flow_frame[index, 1,:,:] = u2 # torch.from_numpy(data)

            im = visualizeFlow(flow_frame[index, :,:,:])
            print(im.shape)
            imsave(path + str(index).zfill(7) + ".tiff", im)

        np.save(path + "flows.npy", flow_frame.numpy())

        return flow_frame


    def __len__(self):
        return len(self.folderList)


    def __getitem__(self, index):
        tube = {'rgb' : torch.zeros((1, 3, 64, 224, 224)), 'of': torch.zeros((1, 2, 64, 224, 224)), 'joints' : np.zeros((17,64,3))}

        # Step 1 - iterate over training list
        with open(self.folderList, 'r') as file:
            sample = list(csv.reader(file))[index]
        
        # try:
        # Step 2 - check if preprocessed data for sample exists
        now = time.time()
        path, h, w = self.prepare(sample, self.frameDir, self.jsonDir)
        print(time.time() - now)
        # tube["scene"] = torch.zeros((1, 3, 64, h, w))    

        
        # Step 3 - Extract Tube & Augment if necessary
        '''
        res = self.extractTube(path, sample, h, w, joint_info)
        if res == "ERR":
            return "ERR"

        tube["rgb"], tube["scene"], tube["joints"], bb = res[0], res[1], res[2], res[3]
        '''

        '''

        # Step 4 - Extract Flow Tube
        #tube["of"] = self.extractFlow(path, tube["scene"], h,  w, bb)
        np.save(path + "/joint/ehpi.npy", tube["joints"])
        # im = tube["joints"].numpy()
        imsave(path + "/joint/ehpi.tiff", tube["joints"])
        # im = Image.fromarray(tube["joints"], "RGB")
        # im.save(path + "/joint/ehpi.tiff")

        # np.transpose(i, (2, 0, 1))
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
        #except Exception as e:
        #    print(sys.exc_info()[-1].tb_lineno)
         #   print("Exception!!!!", e)
         #   return "EXCEPT"
        # tube["of"] = self.extractFlow(path, tube["scene"], h,  w, bb)
        # step 6 - return tube
        return tube