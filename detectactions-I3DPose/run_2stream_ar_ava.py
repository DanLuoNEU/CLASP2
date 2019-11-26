import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import skimage
import os
from torch.utils.data import Dataset, DataLoader
import re
import random
from torchvision.transforms import ToTensor
import csv
from dataloader import ava_dataset
import subprocess
import cv2
import sys
import argparse
from skimage.io import imread, imsave
import json
import videotransforms
from pytorch_i3d import InceptionI3d

# from ehpiClassifier import EHPIClassifier
import datetime
import time

LOGFILE = "./logfile.txt"

def writeLogFile(s):
    try:
        f = open(LOGFILE, "a")
        f.write(str(datetime.datetime.now()) + ":\t" + str(s) + "\n")
        f.close()
        return True
    except Exception as e:
        print(e)
        return False


def freezeWeightsFeatureExtraction(model):
	for name, param in model.module.named_parameters():
		if(param.requires_grad):
			if(not ("logits" in name)):
				param.requires_grad = False
	# train_model(retinanet, dataset_train, dataset_val, dataloader_train,dataloader_val)
	return model

# Random Seed
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

mode = "trainval"
# rootDir = '/storage/truppr/AVA/ava-dataset-tool/'
rootDir = '/data/truppr/AVA/'
videoDir = rootDir + "videos/" 
# frameDir = rootDir + "video/samples/" + mode
jsonDir = rootDir + "streams/"
flowDir = rootDir + "flows/"
# ava_training_set = "ava_dataset_files/ava_train_v2.1.csv"
ava_training_set = "ava_dataset_files/ava_random_train_truppr_v2class.csv"
ava_validation_set = "ava_dataset_files/ava_random_valid_truppr_v2class.csv"

train_data = ava_dataset(ava_training_set, videoDir, flowDir, jsonDir)
valid_data = ava_dataset(ava_validation_set, videoDir, flowDir, jsonDir)

numClasses = 2;

########## Activity Recognition - EHPI Stream
# ehpi_stream = EHPIClassifier(numClasses)
# ehpi_stream.cuda(0)

########## Activity Recognition - RGB Stream
i3d_RGB = InceptionI3d(157, in_channels=3) # 400 when only loaded with imagenet weights
i3d_RGB.load_state_dict(torch.load('models/rgb_charades.pt'))
i3d_RGB.replace_logits(numClasses)
i3d_RGB.cuda(0)
i3d_RGB = nn.DataParallel(i3d_RGB)

########## Activity Recognition - Optical Flow Stream
i3d_OF = InceptionI3d(157, in_channels=2) # 400 when only loaded with imagenet weights
i3d_OF.load_state_dict(torch.load('models/flow_charades.pt'))
i3d_OF.replace_logits(numClasses)
i3d_OF.cuda(0)
i3d_OF = nn.DataParallel(i3d_OF)

# ehpi_stream.train(True);
i3d_RGB.train(True);
i3d_OF.train(True);

lr = 0.01 # consider shrinking for feature extraction
optimizer_rgb = optim.SGD(i3d_RGB.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
optimizer_flow = optim.SGD(i3d_OF.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)

lr_sched_rgb = optim.lr_scheduler.MultiStepLR(optimizer_rgb, [150, 200])
lr_sched_flow = optim.lr_scheduler.MultiStepLR(optimizer_flow, [150, 200])

num_steps_per_update = 4 # accum gradient

i3d_RGB = freezeWeightsFeatureExtraction(i3d_RGB)
i3d_OF = freezeWeightsFeatureExtraction(i3d_OF)

startEpoch = 0
stopEpoch = 100


for epoch in range(startEpoch, stopEpoch):
    writeLogFile("Starting training (EPOCH " + str(epoch) + ")")
    print("Starting training (EPOCH " + str(epoch) + ")")

    error = 0
    num_iter = 0
    steps = 0
    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0

    for tube in train_data:
        # ehpi_stream.train(True);
        i3d_RGB.train(True);
        i3d_OF.train(True);

        i3d_in_rgb = torch.zeros(1,3,64,224,224)
        i3d_in_of = torch.zeros(1,2,64,224,224)

        if tube["action"] == -1:
            print("Skipping tube because of problem...")
            error = error + 1
            continue

        i3d_in_rgb = Variable(tube["rgb"]).cuda(0)
        i3d_in_of = Variable(tube["of"]).cuda(0)

        # Flow Stream
        per_flow_logits = i3d_OF(i3d_in_of)

        # RGB Stream
        per_frame_logits = i3d_RGB(i3d_in_rgb)
        t = i3d_in_rgb.size(2)

        logits = (per_flow_logits + per_frame_logits) / 2.0
        logits = F.upsample(logits, t, mode='linear')

        # labels = Variable(torch.zeros(1, 2, 64).float().cpu(), requires_grad=True)
        labels = torch.zeros(1, 2, 64)
        # labels = createLabel(datum, frame);
        if tube["action"] == "push":
            labels[0, 0, :] = 1
        elif tube["action"] == "pick up":
            labels[0, 1, :] = 1
        else:
            print("ERROR: unrecognized action...", tube["action"])
            input()
        
        labels = Variable(labels).cuda(0)

        # compute localization loss
        # loc_loss = F.binary_cross_entropy_with_logits(logits.detach().cpu(), labels)
        loc_loss = F.binary_cross_entropy_with_logits(logits, labels)
        tot_loc_loss += loc_loss.item()

        # compute classification loss (with max-pooling along time B x C x T)
        # cls_loss = F.binary_cross_entropy_with_logits(torch.max(logits.detach().cpu(), dim=2)[0], torch.max(labels, dim=2)[0])
        cls_loss = F.binary_cross_entropy_with_logits(torch.max(logits, dim=2)[0], torch.max(labels, dim=2)[0])
        tot_cls_loss += cls_loss.item()

        loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
        tot_loss += loss.item()
        loss.backward()

        if num_iter == num_steps_per_update: #  and phase == 'train':
            steps += 1
            num_iter = 0
            optimizer_rgb.step()
            optimizer_flow.step()
            optimizer_rgb.zero_grad()
            optimizer_flow.zero_grad()
            lr_sched_rgb.step()
            lr_sched_flow.step()

            if steps % 10 == 0:
                print('Loc Loss: ' + str(tot_loc_loss/(10*num_steps_per_update)) + ' Cls Loss: ' + str(tot_cls_loss/(10*num_steps_per_update)) + ' Tot Loss: ' + str(tot_loss/10))
                writeLogFile('Loc Loss: ' + str(tot_loc_loss/(10*num_steps_per_update)) + ' Cls Loss: ' + str(tot_cls_loss/(10*num_steps_per_update)) + ' Tot Loss: ' + str(tot_loss/10))
                torch.save(i3d_RGB.module.state_dict(), "rgbStream_"+str(steps).zfill(6)+'-'+str(tot_loss/10)+'.pt')
                torch.save(i3d_OF.module.state_dict(), "floStream_"+str(steps).zfill(6)+'-'+str(tot_loss/10)+'.pt')
                tot_loss = tot_loc_loss = tot_cls_loss = 0.
        num_iter = num_iter + 1;
    
    

    if epoch % 2 == 0:
        error = 0
        num_iter = 0
        steps = 0;
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
        valid_index = 0;
        writeLogFile("Starting VALIDATION")
        for tube in valid_data:
            i3d_RGB.train(False);
            i3d_OF.train(False);
            i3d_RGB.eval()
            i3d_OF.eval()

            i3d_in_rgb = torch.zeros(1,3,64,224,224)
            i3d_in_of = torch.zeros(1,2,64,224,224)

            if tube["action"] == -1:
                print("Skipping tube because of problem...")
                error = error + 1
                continue

            i3d_in_rgb = Variable(tube["rgb"]).cuda(0)
            i3d_in_of = Variable(tube["of"]).cuda(0)

            # Flow Stream
            per_flow_logits = i3d_OF(i3d_in_of)

            # RGB Stream
            per_frame_logits = i3d_RGB(i3d_in_rgb)
            t = i3d_in_rgb.size(2)

            logits = (per_flow_logits + per_frame_logits) / 2.0
            logits = F.upsample(logits, t, mode='linear')

            # labels = Variable(torch.zeros(1, 2, 64).float().cpu(), requires_grad=True)
            labels = torch.zeros(1, 2, 64)
            # labels = createLabel(datum, frame);
            if tube["action"] == "push":
                labels[0, 0, :] = 1
            elif tube["action"] == "pick up":
                labels[0, 1, :] = 1
            else:
                print("ERROR: unrecognized action...", tube["action"])
                input()
            
            labels = Variable(labels).cuda(0)

            # compute localization loss
            # loc_loss = F.binary_cross_entropy_with_logits(logits.detach().cpu(), labels)
            loc_loss = F.binary_cross_entropy_with_logits(logits, labels)
            tot_loc_loss += loc_loss.item()

            # compute classification loss (with max-pooling along time B x C x T)
            # cls_loss = F.binary_cross_entropy_with_logits(torch.max(logits.detach().cpu(), dim=2)[0], torch.max(labels, dim=2)[0])
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(logits, dim=2)[0], torch.max(labels, dim=2)[0])
            tot_cls_loss += cls_loss.item()

            loss = (0.5*loc_loss + 0.5*cls_loss)
            tot_loss += loss.item()
            valid_index = valid_index + 1

        writeLogFile('VALIDATION: Loc Loss: ' + str(tot_loc_loss/(valid_index)) + ' Cls Loss: ' + str(tot_cls_loss/(valid_index)) + ' Tot Loss: ' + str(tot_loss/valid_index))
        print('VALIDATION: Loc Loss: ' + str(tot_loc_loss/(valid_index)) + ' Cls Loss: ' + str(tot_cls_loss/(valid_index)) + ' Tot Loss: ' + str(tot_loss/valid_index))
        torch.save(i3d_RGB.module.state_dict(), "rgbStream_e"+str(epoch)+"-"+str(tot_cls_loss/(valid_index))+'.pt')
        torch.save(i3d_OF.module.state_dict(), "floStream_e"+str(epoch)+"-"+str(tot_cls_loss/(valid_index))+'.pt')