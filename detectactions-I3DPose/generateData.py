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

from ehpiClassifier import EHPIClassifier

import time




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
ava_training_set = "ava_dataset_files/ava_train_truppr_v2class.csv"

train_data = ava_dataset(ava_training_set, videoDir, flowDir, jsonDir)

for tube in train_data:
    if tube["action"] == -1:
        print("Skipping tube because of some problem...")
        error = error + 1
        continue