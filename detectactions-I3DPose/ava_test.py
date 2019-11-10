import torch.utils.data as data
import torch
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
from AVA_dataloader import ava_dataset
import subprocess

mode = "trainval"
rootDir = '/storage/truppr/AVA/ava-dataset-tool/'
videoDir = rootDir + "video/" + mode
frameDir = rootDir + "video/samples/" + mode
jsonDir = rootDir + "json_files/"
flowDir = rootDir + "flows/"
# ava_training_set = "ava_dataset_files/ava_train_v2.1.csv"
ava_training_set = "ava_dataset_files/ava_train_truppr_v9.9.csv"

data = ava_dataset(ava_training_set, videoDir, flowDir, jsonDir)

########## TVNet for Optical Flow Extraction for I3D Flow Stream
'''
from model.network import model as tvnet
tvnet_args = arguments().parse() # tvnet parameters
tvnet_args.batch_size = 4; # 3 + 1 for UCF101
tvnet_args.data_size = [tvnet_args.batch_size, 3, 240, 320]
init_file = '/home/truppr/kDYAN-TV/savedModels/UCF-DYAN-TVNET-1-1-30-E2E-DL_1_0.3163.pth'
of = tvnet(tvnet_args).cuda()
of = loadTVNet(of, init_file)
of = of.cpu()
'''

for tube in data:
    pass
    # print(tube)
    # print(prapre(tube[0]+".mkv", frameDir, jsonDir, 901, 903))
    # input()