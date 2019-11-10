import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import cv2
import sys
from progress.bar import Bar
import random
import json

import argparse
from PIL import Image
import skimage
from skimage.io import imread, imsave
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from custom_dataloaders import clasp_frameset
import videotransforms

from custom_dataloaders import clasp_sliding_window_frameset

from tvnet1130_train_options import arguments
from model.network import model as tvnet

import numpy as np
from pytorch_i3d import InceptionI3d
import time

### TRAIN or TEST !!!
_TRAIN = True # False;


# Random Seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

# Test Data
# trainingData = { "cam22exp1" : ["person.json"] }

# CLASP Data Combo
trainingData = { "cam11exp4a" : ["xfr-from-576", "xfr-from-660", "xfr-to-233", "xfr-to-520"], "cam11exp5a" : ["xfr-from-1204", "xfr-from-1296", "xfr-from-1512", "xfr-from-1742", "xfr-from-1932", "xfr-from-2213", "xfr-from-2289", "xfr-from-2409", "xfr-from-2493", "xfr-from-2611", "xfr-from-2796", "xfr-from-2912", "xfr-from-3063", "xfr-from-3075"], "cam11exp6a" : ["xfr-from-1272", "xfr-from-1337", "xfr-from-1485", "xfr-from-1535", "xfr-from-1575", "xfr-from-1640"], "cam13exp5a" : ["xfr-from-2793", "xfr-from-3112"], "cam13exp6a" : ["xfr-from-1582", "xfr-from-1599"], "cam9exp1a" : ["xfr-from-638", "xfr-from-700", "xfr-to-331", "xfr-to-592"], "cam9exp2a" : ["xfr-from-493", "xfr-from-566", "xfr-to-233", "xfr-to-433"], "cam9exp3a" : ["xfr-from-705", "xfr-from-745", "xfr-to-260", "xfr-to-651"], "cam9exp5a" : ["xfr-to-1166", "xfr-to-1248", "xfr-to-1422", "xfr-to-1697", "xfr-to-1725", "xfr-to-1928", "xfr-to-1985", "xfr-to-2036", "xfr-to-2085", "xfr-to-2209", "xfr-to-2355", "xfr-to-2643", "xfr-to-617", "xfr-to-700", "xfr-to-875", "xfr-to-943"], "cam9exp6a" : ["xfr-to-1076", "xfr-to-1190", "xfr-to-1375", "xfr-to-466", "xfr-to-511", "xfr-to-551", "xfr-to-567", "xfr-to-702", "xfr-to-746", "xfr-to-924"], "cam09exp1" : ['xfr-to-1350', 'xfr-to-1460', 'xfr-to-1620', 'xfr-to-1710', 'xfr-to-1860', 'xfr-to-2540', 'xfr-to-2600', 'xfr-to-2610', 'xfr-to-2680', 'xfr-to-2710', 'xfr-to-6250', 'xfr-to-6290', 'xfr-to-7100', 'xfr-to-7360', 'xfr-to-7470', 'xfr-to-950'], "cam11exp1" : ['xfr-from-2097', 'xfr-from-2822', 'xfr-from-2895', 'xfr-from-3230', 'xfr-from-3444', 'xfr-from-3496', 'xfr-from-3647', 'xfr-from-4244', 'xfr-from-4414', 'xfr-from-5937', 'xfr-from-6772', 'xfr-from-7049', 'xfr-from-7086', 'xfr-from-7190', 'xfr-from-7250', 'xfr-from-7306', 'xfr-from-7556', 'xfr-from-8063', 'xfr-from-8175', 'xfr-from-8267', 'xfr-from-9769']}

# CLASP 1 Data
# trainingData = { "cam11exp4a" : ["xfr-from-576", "xfr-from-660", "xfr-to-233", "xfr-to-520"], "cam11exp5a" : ["xfr-from-1204", "xfr-from-1296", "xfr-from-1512", "xfr-from-1742", "xfr-from-1932", "xfr-from-2213", "xfr-from-2289", "xfr-from-2409", "xfr-from-2493", "xfr-from-2611", "xfr-from-2796", "xfr-from-2912", "xfr-from-3063", "xfr-from-3075"], "cam11exp6a" : ["xfr-from-1272", "xfr-from-1337", "xfr-from-1485", "xfr-from-1535", "xfr-from-1575", "xfr-from-1640"], "cam13exp5a" : ["xfr-from-2793", "xfr-from-3112"], "cam13exp6a" : ["xfr-from-1582", "xfr-from-1599"], "cam9exp1a" : ["xfr-from-638", "xfr-from-700", "xfr-to-331", "xfr-to-592"], "cam9exp2a" : ["xfr-from-493", "xfr-from-566", "xfr-to-233", "xfr-to-433"], "cam9exp3a" : ["xfr-from-705", "xfr-from-745", "xfr-to-260", "xfr-to-651"], "cam9exp5a" : ["xfr-to-1166", "xfr-to-1248", "xfr-to-1422", "xfr-to-1697", "xfr-to-1725", "xfr-to-1928", "xfr-to-1985", "xfr-to-2036", "xfr-to-2085", "xfr-to-2209", "xfr-to-2355", "xfr-to-2643", "xfr-to-617", "xfr-to-700", "xfr-to-875", "xfr-to-943"], "cam9exp6a" : ["xfr-to-1076", "xfr-to-1190", "xfr-to-1375", "xfr-to-466", "xfr-to-511", "xfr-to-551", "xfr-to-567", "xfr-to-702", "xfr-to-746", "xfr-to-924"] }

# CLASP 2 Data
# trainingData = { "cam09exp1" : ['xfr-to-1350', 'xfr-to-1460', 'xfr-to-1620', 'xfr-to-1710', 'xfr-to-1860', 'xfr-to-2540', 'xfr-to-2600', 'xfr-to-2610', 'xfr-to-2680', 'xfr-to-2710', 'xfr-to-6250', 'xfr-to-6290', 'xfr-to-7100', 'xfr-to-7360', 'xfr-to-7470', 'xfr-to-950'], "cam11exp1" : ['xfr-from-2097', 'xfr-from-2822', 'xfr-from-2895', 'xfr-from-3230', 'xfr-from-3444', 'xfr-from-3496', 'xfr-from-3647', 'xfr-from-4244', 'xfr-from-4414', 'xfr-from-5937', 'xfr-from-6772', 'xfr-from-7049', 'xfr-from-7086', 'xfr-from-7190', 'xfr-from-7250', 'xfr-from-7306', 'xfr-from-7556', 'xfr-from-8063', 'xfr-from-8175', 'xfr-from-8267', 'xfr-from-9769']}



if not os.path.exists('output'):
	os.mkdir('output')

# start = time.time()
# print("starting...")
# windowIndex = 0;

########## Load Actions
action = open("./action_list.txt").readlines()
excluded_actions = []
idx = 0;
for act in action:
	if act.split(' ')[0] == str(0):
		excluded_actions.append(act[2:].replace('\n',''))

	action[idx] = act[2:].replace('\n','')# .join(' ')	
	idx = idx + 1;

# print(action)

# print(excluded_actions)
	

# ar_input = torch.zeros(1,3,64,224,224)
# ar_flow = torch.zeros(1,2,64,224,224)
# tf = transforms.Compose([videotransforms.CenterCrop(224)])
tf1 = torchvision.transforms.Resize((240,320))
tf2 = torchvision.transforms.Resize((224,224))


########## Local Funcs
def frame_to_tensor(frame, hf):
	# img_path1 = os.path.join(path, str('%07d' % framenum) + '.jpg')       
	image1 = Image.open(frame)
	image1 = tf1(image1)

	if hf:
		image1 = torchvision.transforms.RandomHorizontalFlip(p=1)(image1)

	image1 = ToTensor()(image1)

	# image1 = image1.double()
	# data = np.transpose(image1.numpy(), (1, 2, 0)) # put height and width in front

	# data = skimage.transform.resize(data, (240, 320))

	
	# print("local func image1: ", image1.shape)
	# image1 = torch.from_numpy(np.transpose(data, (2, 0, 1) ) ) # move back 
	# frames[:,i,:] = image1 # .view(3, 480 * 640)    
	return image1

def viableActionTube(imgs):
	previous = 0
	count = 0;
	for im in imgs:
		# print(im)
		# print(int(im.replace(".jpg",'')) - previous)
		if int(im.replace(".jpg",'')) - previous < 10:
			count = count + 1;
		else:
			if count == 0:
				pass
			else:
				# print(imgs)
				# print()
				# print(im)
				print("Interruption in tube!!!")
			count = 0;
			# continue
		previous = int(im.replace(".jpg",''))
	# print()

	if count >= 63:
		return True
	else:
		return False


def convertBB(yfrom, xfrom, yto, xto, bbox, hf, sbb, shift_x, shift_y):
	arr = [0, 0, 0, 0]

	# Flip if need be
	if hf:
		arr[0] = xfrom - arr[0]
		arr[1] = arr[1]
		arr[2] = xfrom - arr[2]
		arr[3] = arr[3]

	# Shift if need be
	'''
	shift_x = 0
	shift_y = 0
	if sbb:
		shift_x = random.choice([i for i in range(-1 * int((bbox[2] - bbox[0]) / 4.0), int((bbox[2] - bbox[0]) / 4.0))])
		shift_y = random.choice([i for i in range(-1 * int((bbox[3] - bbox[1]) / 4.0), int((bbox[3] - bbox[1]) / 4.0))])
		print("Shifting Data! (x, y): (", shift_x, ",",shift_y,")")
	'''

	# Enlarge the BB
	arr[0] = bbox[0] - ((bbox[2] - bbox[0]) / 4.0) + shift_x
	arr[1] = bbox[1] - ((bbox[3] - bbox[1]) / 4.0) + shift_y
	arr[2] = bbox[2] + ((bbox[2] - bbox[0]) / 4.0) + shift_x
	arr[3] = bbox[3] + ((bbox[3] - bbox[1]) / 4.0) + shift_y
	
	# Scale the BB
	arr[0] = int((arr[0] / xfrom) * xto)
	arr[1] = int((arr[1] / yfrom) * yto)
	arr[2] = int((arr[2] / xfrom) * xto)
	arr[3] = int((arr[3] / yfrom) * yto)
	arr = [0 if i < 0 else i for i in arr]
	
	# Clip the BB
	if arr[1] > yto:
		arr[1] = yto - 1;
	if arr[3] > yto:
		arr[3] = yto - 1;
	if arr[0] > xto:
		arr[0] = xto - 1;
	if arr[2] > xto:
		arr[2] = xto - 1;

	return arr


def convertFrame(f):
	# print(f)
	new_frame = str(int(f.replace('.jpg','')) - 32).zfill(7) + ".jpg"
	# print(new_frame)
	# input()
	return new_frame


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


def visualizeFlow(f):
    flow = np.zeros((224, 224, 2))
    flow[:, :, 0] = f[0,:,:]# .cpu().numpy()
    flow[:, :, 1] = f[1,:,:]# .cpu().numpy()
    hsv = np.zeros((224, 224, 3), dtype=np.uint8)

    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = bgr[...,::-1] # comment out when saving with cv2.imwrite

    return rgb


def maxAcceptedAction(activities, action):
	val, idx = torch.max(activities_at_midpoint, dim=0)
	while action[idx] in excluded_actions:
		activities[idx] = -1;
		val, idx = torch.max(activities, dim=0)

	return val, idx

def stripPersonsJSON(pose_data, entry, datum):
	fr = str(datum.split('-')[2]).zfill(7) + ".jpg"
	action = datum.split('-')[1]

	image_path = '/storage/truppr/CLASP-Activity-Dataset/' + entry
	
	possibles = []

	return_dict = {}

	# print("Looking at ", entry, " for action ", action, " at frame ", fr)
	if os.path.isfile(outputDir + "output-"+str(fr)):
		orig_image = imread(outputDir + "output-"+str(fr))
	else:
		orig_image = imread(image_path + "/" + fr)

	# orig_image = Image.open(image_path + "/" + fr) 
	for pers in pose_data:
		try:
			bbox = pose_data[pers]["bbox"][pose_data[pers]["img_name"].index(str(fr))]
			print("person ", pers, " is the one who matters...", " w/ len: ", len(pose_data[pers]["bbox"]))
			possibles.append(pers)
			label = "person " + str(pers)
			cv2.rectangle(orig_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 4)
			cv2.putText(orig_image, label, (int(bbox[0]) + 20, int(bbox[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
			imsave(outputDir + "output-"+str(fr), orig_image)
		except ValueError:
			continue
	
	# Uncomment to strip useless data from training samples
	print("Which person should be saved for ", str(fr))
	x = input("hold up -> ")	
	return_dict[str(x)] = pose_data[str(x)]

	return return_dict

def freezeWeightsFeatureExtraction(model):
	for name, param in model.module.named_parameters():
		if(param.requires_grad):
			if(not ("logits" in name)):
				param.requires_grad = False
	# train_model(retinanet, dataset_train, dataset_val, dataloader_train,dataloader_val)
	return model


def createLabel(sampleName, currentFrame):
	label = torch.zeros(1, 2, 64) # .float().cpu(), requires_grad=True)

	mid = int(sampleName.split('-')[2])
	# print(mid)
	# print(currentFrame)
	# print(int(currentFrame) - int(mid))

	midpoint = int(sampleName.split('-')[2]) 

	if "from" in sampleName:
		label[0,1, 63 - midpoint - 20: 63 - midpoint + 20] = 1;
	elif "to" in sampleName:
		label[0,0, 63 - midpoint - 20: 63 - midpoint + 20] = 1;
	else:
		input("unknown action sample!!! (", sampleName, ")")

	return Variable(label.float().cpu(), requires_grad=True)

########## TVNet for Optical Flow Extraction for I3D Flow Stream
from model.network import model as tvnet
tvnet_args = arguments().parse() # tvnet parameters
tvnet_args.batch_size = 4; # 3 + 1 for UCF101
tvnet_args.data_size = [tvnet_args.batch_size, 3, 240, 320]
# init_file = '/home/truppr/kDYAN-TV/savedModels/UCF-TVNet-1-1-30.pth'
init_file = '/home/truppr/kDYAN-TV/savedModels/UCF-DYAN-TVNET-1-1-30-E2E-DL_1_0.3163.pth'
of = tvnet(tvnet_args).cuda()

of = loadTVNet(of, init_file)

of = of.cpu()


########## Activity Recognition - RGB Stream
i3d_RGB = InceptionI3d(157, in_channels=3) # 400 when only loaded with imagenet weights
i3d_RGB.load_state_dict(torch.load('models/rgb_charades.pt'))
i3d_RGB.replace_logits(2)
i3d_RGB.cuda(0)
i3d_RGB = nn.DataParallel(i3d_RGB)


########## Activity Recognition - Optical Flow Stream
i3d_OF = InceptionI3d(157, in_channels=2) # 400 when only loaded with imagenet weights
i3d_OF.load_state_dict(torch.load('models/flow_charades.pt'))
i3d_OF.replace_logits(2)
i3d_OF.cuda(0)
i3d_OF = nn.DataParallel(i3d_OF)

if _TRAIN:
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

else:
	i3d_RGB.train(False)
	i3d_OF.train(False)

start_epoch = 0;
max_epoch = 20;
######### Main Loop
for epoch in range(start_epoch, max_epoch):
	action = ["Divesting", "Revesting"]
	num_iter = 0
	steps = 0;
	tot_loss = 0.0
	tot_loc_loss = 0.0
	tot_cls_loss = 0.0

	for entry in trainingData:
		for datum in trainingData[entry]:
			# print("Entry: ", entry)
			# print(trainingData[entry])

			# print("Looking at: ", datum)
			camera = '/' + entry +  '/'
			rootDir = '/storage/truppr/CLASP-Activity-Dataset/'
			flowDir = '/storage/truppr/CLASP-Activity-Dataset/flows/' + camera
			image_path = '/storage/truppr/CLASP-Activity-Dataset/' + camera
			outputDir = './output/' + camera
			# /storage/truppr/CLASP-Activity-Dataset/CLASP-Activity-Dataset/person_info/
		
			if _TRAIN:
				personTracker = rootDir + "CLASP-Activity-Dataset/person_info/" + camera + "/" + datum + "/persons.json"
			else:
				personTracker = rootDir + "CLASP-Activity-Dataset/person_info/" + camera + "/persons.json"

			# load Dan json file
			people = 0
			with open(personTracker) as json_file:
				pose_data = json.load(json_file)
				# for entry in pose_data:
				# 	print(pose_data[entry]["img_name"])
				#	people = people + 1
			# input(pose_data[entry])

			# Training Data Assembly...
			'''
			pose_data = stripPersonsJSON(pose_data, entry, datum)
			json.dump(pose_data, open(personTracker, "w"))
			continue				
			'''

			# tot_loss = 0.0
			# tot_loc_loss = 0.0
			# tot_cls_loss = 0.0
			# num_iter = 0
			# steps = 0;

			ar_input = torch.zeros(1,3,64,224,224)
			ar_flow = torch.zeros(1,2,64,224,224)

			# Parse persons.json
			person_found = False;
			for pers in pose_data:
				print("Person ", pers, "-> length ", len(pose_data[pers]["img_name"]))
				if viableActionTube(pose_data[pers]["img_name"]):
					print("\t(Present in enough frames for action recognition!!!)")
					hf = False;
					sbb = False;
					shift_x = 0;
					shift_y = 0;
					if _TRAIN and random.choice([0, 1]):
						print("Flipping Data!!!")
						hf = True;

					''' Don't do shifts...
					if _TRAIN and random.choice([0, 1]):
						print("Shifting Data!")
						sbb = True
						shift_x = random.choice([i for i in range(-25,25)])
						shift_y = random.choice([i for i in range(-25,25)])
						print("shifting (x, y): (", shift_x, ",",shift_y,")")
					'''

					previousFrame = 0;
					windowIndex = 0;
					
					if len(pose_data[pers]["img_name"]) < 63:
						windowIndex = 63 - len(pose_data[pers]["img_name"])

					i3d_in_rgb = torch.zeros(1,3,64,224,224)
					i3d_in_of = torch.zeros(1,2,64,224,224)
					for fr in pose_data[pers]["img_name"]:
						if int(fr.replace(".jpg",'')) - previousFrame > 10:
							print("Skipping frame", fr)
							previousFrame = int(fr.replace(".jpg",''))
							previousIm = fr;

							continue;
						elif int(fr.replace(".jpg",'')) - previousFrame == 1:
							# print("Nomral transition...")
							pass
						else:
							frame_shift = int(fr.replace(".jpg",'')) - previousFrame
						
							# update buffers...
							if frame_shift + windowIndex > 63:
								# print("loopoing...")
								for loop in range(0, frame_shift + windowIndex - 63):         
									for i in range(0, 63):
										i3d_in_rgb[0,:,i,:,:] = i3d_in_rgb[0,:,i + 1,:,:]
										i3d_in_of[0,:,i,:,:] = i3d_in_of[0,:,i + 1,:,:]
									i3d_in_rgb[0,:,63,:,:] = torch.zeros(3, 224, 224)
									i3d_in_of[0,:,63,:,:] = torch.zeros(2, 224, 224)
	
								windowIndex = 63 #
							else:
								windowIndex = windowIndex + frame_shift - 1

						print("Processing Frame ", fr, " for person ", pers)

						### Load CLASP frameset
						# orig_image = cv2.imread(image_path+fr)

						if not hf:
							orig_image = Image.open(image_path+fr)
						else:
							orig_image = Image.open(image_path+fr)
							orig_image = torchvision.transforms.RandomHorizontalFlip(p=1)(orig_image)
						# print(orig_image.shape)

						# image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
						# image = skimage.transform.resize(image, (240, 320))
						# image = tf1(orig_image)
						frame =  fr.replace('.jpg', '').replace('.jpeg','')
						flow_frame = torch.zeros(2,240,320)
	

						### Extract Optical Flow from frames
			
						x2 = frame_to_tensor(image_path+fr, hf).type(torch.FloatTensor) # .cuda();
						image = x2;
						if (_TRAIN) or (not _TRAIN and not os.path.isfile(flowDir + str(fr).replace('.jpg','') + "-frame.npy")):
							x1 = frame_to_tensor(image_path+previousIm, hf).type(torch.FloatTensor) # .cuda();
							# x2 = frame_to_tensor(image_path+fr).type(torch.FloatTensor) # .cuda();
							image = x2;

							u1, u2 = of(x2.unsqueeze(0), x1.unsqueeze(0), need_result=True)
			
							# print(u1.shape)

							flow_frame[0,:,:] = u1.detach();
							flow_frame[1,:,:] = u2.detach();
				
							if not os.path.exists(flowDir):
								os.mkdir(flowDir)

							if (not _TRAIN):
								np.save(flowDir + str(fr).replace('.jpg','') + "-frame.npy", flow_frame.numpy())	
						else:
							print("Skipping flow generation!!!")
							flow_frame = torch.from_numpy(np.load(flowDir + str(fr).replace('.jpg','') + "-frame.npy"))

						### Extract RGB BB from persons.json
						bbox = pose_data[pers]["bbox"][pose_data[pers]["img_name"].index(str(fr))]

						### Resize OF and RGB			
						_bbox = convertBB(1080, 1920, 240, 320, bbox, hf, sbb, shift_x, shift_y)
						# print("bbox: ", bbox)
						# print("_bbox: ", _bbox) 			

						# image = tf2(image)
						# image = ToTensor()(image).numpy()
						# print("image size: ", image.shape)
						image = skimage.transform.resize(image.numpy(), (3, 240, 320))
						# print("image size: ", image.shape)
						image = image[:,_bbox[1]:_bbox[3],_bbox[0]:_bbox[2]]
						# print("image size: ", image.shape)
						image = skimage.transform.resize(image, (3, 224, 224))
						# print("image size: ", image.shape)
						i3d_in_rgb[0,:,windowIndex,:,:] = Variable(torch.from_numpy(image))

						flow_frame = skimage.transform.resize(flow_frame.numpy()[:,_bbox[1]:_bbox[3],_bbox[0]:_bbox[2]], (2, 224, 224)) 
						i3d_in_of[0,:,windowIndex,:,:] = Variable(torch.from_numpy(flow_frame))


						### Activity Recognition
						activities_at_midpoint = torch.zeros(3)
						if windowIndex == 63:
							# print("Caught enough action to assemble an action tube!")
							# input()
							inputs = Variable(i3d_in_rgb).cuda()
							flow_inputs = Variable(i3d_in_of).cuda()

							# Flow Stream
							per_flow_logits = i3d_OF(flow_inputs)

							# RGB Stream
							per_frame_logits = i3d_RGB(inputs)
							t = inputs.size(2)

							logits = (per_flow_logits + per_frame_logits) / 2.0
							logits = F.upsample(logits, t, mode='linear')
				
							if _TRAIN:
								# Create Labels...
								labels = Variable(torch.zeros(1, 2, 64).float().cpu(), requires_grad=True)
								labels = createLabel(datum, frame);


								# compute localization loss
								loc_loss = F.binary_cross_entropy_with_logits(logits.detach().cpu(), labels)
								tot_loc_loss += loc_loss.item()
					
								# compute classification loss (with max-pooling along time B x C x T)
								cls_loss = F.binary_cross_entropy_with_logits(torch.max(logits.detach().cpu(), dim=2)[0], torch.max(labels, dim=2)[0])
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
										torch.save(i3d_RGB.module.state_dict(), "rgbStream_"+str(steps).zfill(6)+'-'+str(tot_loss/10)+'.pt')
										torch.save(i3d_OF.module.state_dict(), "floStream_"+str(steps).zfill(6)+'-'+str(tot_loss/10)+'.pt')
										tot_loss = tot_loc_loss = tot_cls_loss = 0.
	
							else:
								# upsample to input size
								# per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')
								# per_flow_logits = F.upsample(per_flow_logits, t, mode='linear')
								# print(per_frame_logits)	
								# sm_rgb = F.softmax(per_frame_logits, dim=2)
								sm = F.softmax(logits, dim=2)
								print(sm)
								# sm = (sm_rgb + sm_flo)/2.0
								activities_at_midpoint = sm[0,:,32]

								# print(sm_rgb.shape)
								# print(sm_flo.shape)
								# print(sm.shape)
								'''
								torch.set_printoptions(profile="full")
								print(activities_at_midpoint)
								torch.set_printoptions(profile="default")
								'''
								# val, idx = torch.max(activities_at_midpoint, dim=0)
								# high_score = torch.max(val, dim=0)

								val, idx = maxAcceptedAction(activities_at_midpoint, action)

								print("action: ", action[idx], " Prob: ", val, " Frame: ", convertFrame(fr).replace('.jpg',''))					
							
								### Visualize Activity Recognition	
								ima = cv2.imread(outputDir + "output-"+convertFrame(fr))
								label = str(action[idx])
								old_bbox = pose_data[pers]["bbox"][pose_data[pers]["img_name"].index(str(fr)) - 32]
								# cv2.putText(ima, label, (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)	
								cv2.putText(ima, label, (int(old_bbox[0]) + 20, int(old_bbox[1]) + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2) 
								cv2.imwrite(outputDir + "output-"+convertFrame(fr), ima)

							# update buffers...
							for i in range(0, 63):
								i3d_in_rgb[0,:,i,:,:] = i3d_in_rgb[0,:,i + 1,:,:]
								i3d_in_of[0,:,i,:,:] = i3d_in_of[0,:,i + 1,:,:]
							i3d_in_rgb[0,:,63,:,:] = torch.zeros(3, 224, 224)
							i3d_in_of[0,:,63,:,:] = torch.zeros(2, 224, 224)

							windowIndex = windowIndex - 1	
							num_iter = num_iter + 1;

						### Visualize Frames...
						if os.path.isfile(outputDir +  "output-"+str(fr)):
							# orig_image = cv2.imread(outputDir + "output-"+str(fr))
							orig_image = imread(outputDir + "output-"+str(fr))
						else:
							# orig_image = cv2.imread(image_path+fr)
							orig_image = imread(image_path+fr)

						if not os.path.exists(outputDir):
							os.mkdir(outputDir)

						box = bbox
						cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
						label = "Person " + pers
						cv2.putText(orig_image, label, (int(box[0]) + 20, int(box[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
						imsave(outputDir + "output-"+str(fr), orig_image)
						imsave(outputDir + "output-"+str(fr).replace('.jpg','') + "-flow-pers" + str(pers) + ".jpg", visualizeFlow(flow_frame))	
						imsave(outputDir + "output-"+str(fr).replace('.jpg','') + "-rgb" + str(pers) + ".jpg", torch.from_numpy(image).permute(1,2,0).numpy())

						### End Loop
						windowIndex = windowIndex + 1
						previousFrame = int(fr.replace(".jpg",''))
						previousIm = fr;
	
	torch.save(i3d_RGB.module.state_dict(), "rgbStream_e"+str(epoch)+'-'+str(tot_loss/10)+'.pt')
	torch.save(i3d_OF.module.state_dict(), "floStream_e"+str(epoch)+'-'+str(tot_loss/10)+'.pt')

