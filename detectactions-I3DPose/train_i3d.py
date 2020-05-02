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
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer

from custom_dataloaders import clasp_sliding_window_frameset

from tvnet1130_train_options import arguments
from model.network import model as tvnet

import numpy as np
from pytorch_i3d import InceptionI3d
import time

# Random Seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

rootDir = '/storage/truppr/CLASP-Activity-Dataset/'
flowDir = '/storage/truppr/CLASP-Activity-Dataset/flows/cam22exp1/'
image_path = '/storage/truppr/CLASP-Activity-Dataset/cam22exp1/'
outputDir = './output/cam22exp1/'
personTracker = rootDir + "persons.json"

if not os.path.exists('output'):
	os.mkdir('output')

img_list = [x for x in os.listdir(image_path)]
img_list.sort()

start = time.time()
print("starting...")
windowIndex = 0;

########## Load Actions
action = open("./action_list.txt").readlines()
excluded_actions = []
idx = 0;
for act in action:
	if act.split(' ')[0] == str(0):
		excluded_actions.append(act[2:].replace('\n',''))

	action[idx] = act[2:].replace('\n','')# .join(' ')	
	idx = idx + 1;

print(action)

print(excluded_actions)
	
'''
for a in action:
	print(a)
	input()
'''

ar_input = torch.zeros(1,3,64,224,224)
ar_flow = torch.zeros(1,2,64,224,224)
# tf = transforms.Compose([videotransforms.CenterCrop(224)])
tf1 = torchvision.transforms.Resize((240,320))
tf2 = torchvision.transforms.Resize((224,224))
previous_frame = '';


########## Local Funcs
def frame_to_tensor(frame):
	# img_path1 = os.path.join(path, str('%07d' % framenum) + '.jpg')       
	image1 = Image.open(frame)
	image1 = tf1(image1)
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


def convertBB(yfrom, xfrom, yto, xto, bbox):
	arr = [0, 0, 0, 0]

	# Enlarge the BB
	arr[0] = bbox[0] - ((bbox[2] - bbox[0]) / 2.0)
	arr[1] = bbox[1] - ((bbox[3] - bbox[1]) / 2.0)
	arr[2] = bbox[2] + ((bbox[2] - bbox[0]) / 2.0)
	arr[3] = bbox[3] + ((bbox[3] - bbox[1]) / 2.0)
	# print(arr)	
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


def maxAcceptedAction(activities):
	val, idx = torch.max(activities_at_midpoint, dim=0)
	while action[idx] in excluded_actions:
		activities[idx] = -1;
		val, idx = torch.max(activities, dim=0)

	return val, idx

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
i3d_RGB = InceptionI3d(400, in_channels=3)
i3d_RGB.load_state_dict(torch.load('models/rgb_imagenet.pt'))
i3d_RGB.replace_logits(157)
i3d_RGB.cuda(0)
i3d_RGB = nn.DataParallel(i3d_RGB)
i3d_RGB.train(True)


########## Activity Recognition - Optical Flow Stream
i3d_OF = InceptionI3d(400, in_channels=2)
i3d_OF.load_state_dict(torch.load('models/flow_imagenet.pt'))
i3d_OF.replace_logits(157)
i3d_OF.cuda(0)
i3d_OF = nn.DataParallel(i3d_OF)
i3d_OF.train(True)


lr = 0.1
optimizer_rgb = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
optimizer_flow = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)

lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

num_steps_per_update = 4 # accum gradient

######### Main Loop

# load Dan json file
people = 0
with open(personTracker) as json_file:
	pose_data = json.load(json_file)
	# for entry in pose_data:
	# 	print(pose_data[entry]["img_name"])
	#	people = people + 1
# input(pose_data[entry])

screen_play = {}
screen_play["people"] = {}
screen_play["tubes_s"] = {}
screen_play["tubes_t"] = {}

# Parse persons.json
person_found = False;
for pers in pose_data:
	print("Person ", pers, "-> length ", len(pose_data[pers]["img_name"]))
	if viableActionTube(pose_data[pers]["img_name"]):
		print("\t(Present in enough frames for action recognition!!!)")

		previousFrame = 0;
		windowIndex = 0;
		i3d_in_rgb = torch.zeros(1,3,64,224,224)
		i3d_in_of = torch.zeros(1,2,64,224,224)
		for fr in pose_data[pers]["img_name"]:
			if int(fr.replace(".jpg",'')) - previousFrame > 2:
				print("Skipping frame", fr)
				previousFrame = int(fr.replace(".jpg",''))
				previousIm = fr;

				continue;

			print("Processing Frame ", fr, " for person ", pers)
			

			### Load CLASP frameset
			# orig_image = cv2.imread(image_path+fr)
			orig_image = Image.open(image_path+fr)
			# print(orig_image.shape)

			# image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
			# image = skimage.transform.resize(image, (240, 320))
			# image = tf1(orig_image)
			frame =  fr.replace('.jpg', '').replace('.jpeg','')
			flow_frame = torch.zeros(2,240,320)
	

			### Extract Optical Flow from frames
			
			x2 = frame_to_tensor(image_path+fr).type(torch.FloatTensor) # .cuda();
			image = x2;
			if not os.path.isfile(flowDir + str(fr).replace('.jpg','') + "-frame.npy"):	
				x1 = frame_to_tensor(image_path+previousIm).type(torch.FloatTensor) # .cuda();
				# x2 = frame_to_tensor(image_path+fr).type(torch.FloatTensor) # .cuda();
				image = x2;

				u1, u2 = of(x2.unsqueeze(0), x1.unsqueeze(0), need_result=True)
			
				# print(u1.shape)

				flow_frame[0,:,:] = u1.detach();
				flow_frame[1,:,:] = u2.detach();
				
				np.save(flowDir + str(fr).replace('.jpg','') + "-frame.npy", flow_frame.numpy())	
			else:
				print("Skipping flow generation!!!")
				flow_frame = torch.from_numpy(np.load(flowDir + str(fr).replace('.jpg','') + "-frame.npy"))

			### Extract RGB BB from persons.json
			bbox = pose_data[pers]["bbox"][pose_data[pers]["img_name"].index(str(fr))]

			### Resize OF and RGB			
			_bbox = convertBB(1080, 1920, 240, 320, bbox)
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
			activities_at_midpoint = torch.zeros(157)
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

				# combnie output
				logits = (per_flow_logits + per_frame_logits) / 2.0

				# upsample to input size
				per_frame_logits = F.upsample(logits, t, mode='linear')
				per_flow_logits = F.upsample(logits, t, mode='linear')

				# compute localization loss
				loc_loss_rgb = F.binary_cross_entropy_with_logits(logits, labels)
				tot_loc_loss += loc_loss.item()

				# compute classification loss (with max-pooling along time B x C x T)
				cls_loss_rgb = F.binary_cross_entropy_with_logits(torch.max(logits, dim=2)[0], torch.max(labels, dim=2)[0])
				tot_cls_loss += cls_loss.item()
			
				loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
				tot_loss += loss.item()
				loss.backward()

				
				if num_iter == num_steps_per_update and phase == 'train':
					steps += 1
					num_iter = 0
					optimizer.step()
					optimizer.zero_grad()
					lr_sched.step()
					if steps % 10 == 0:
						print(str(phase) + ' Loc Loss: ' + str(tot_loc_loss/(10*num_steps_per_update)) + ' Cls Loss: ' + str(tot_cls_loss/(10*num_steps_per_update)) + ' Tot Loss: ' + str(tot_loss/10))
						torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'-'+str(tot_loss/10)+'.pt')
						tot_loss = tot_loc_loss = tot_cls_loss = 0.

				# sm_rgb = F.softmax(per_frame_logits, dim=2)
				# sm_flo = F.softmax(per_flow_logits, dim=2)
	
				# sm = (sm_rgb + sm_flo)/2.0
				# activities_at_midpoint = sm[0,:,32]

				'''
				torch.set_printoptions(profile="full")
				print(activities_at_midpoint)
				torch.set_printoptions(profile="default")
				'''
				# val, idx = torch.max(activities_at_midpoint, dim=0)
				# high_score = torch.max(val, dim=0)

				val, idx = maxAcceptedAction(activities_at_midpoint)

				print("action: ", action[idx], " Prob: ", val, " Frame: ", convertFrame(fr).replace('.jpg',''))					

				### Visualize Activity Recognition	
				ima = cv2.imread(outputDir + "output-"+convertFrame(fr))
				label = str(action[idx])
				# cv2.putText(ima, label, (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)	
				cv2.putText(ima, label, (int(bbox[0]) + 20, int(bbox[1]) + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2) 
				cv2.imwrite(outputDir + "output-"+convertFrame(fr), ima)

				# update buffers...
				for i in range(0, 63):
					i3d_in_rgb[0,:,i,:,:] = i3d_in_rgb[0,:,i + 1,:,:]
					i3d_in_of[0,:,i,:,:] = i3d_in_of[0,:,i + 1,:,:]
				i3d_in_rgb[0,:,63,:,:] = torch.zeros(3, 224, 224)
				i3d_in_of[0,:,63,:,:] = torch.zeros(2, 224, 224)

				windowIndex = windowIndex - 1	


			### Visualize Frames...
			# print('\t' + str(screen_play["people"][str(peep)]))
			if os.path.isfile(outputDir +  "output-"+str(fr)):
				# orig_image = cv2.imread(outputDir + "output-"+str(fr))
				orig_image = imread(outputDir + "output-"+str(fr))
			else:
				# orig_image = cv2.imread(image_path+fr)
				orig_image = imread(image_path+fr)

			box = bbox
			cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
			#label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
			label = "Person " + pers
			cv2.putText(orig_image, label, (int(box[0]) + 20, int(box[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
			imsave(outputDir + "output-"+str(fr), orig_image)
			# cv2.imwrite(outputDir + "output-"+str(fr), orig_image)

			print("Shapes: ", orig_image.shape, torch.from_numpy(image).permute(1,2,0).numpy().shape)

			imsave(outputDir + "output-"+str(fr).replace('.jpg','') + "-flow-pers" + str(pers) + ".jpg", visualizeFlow(flow_frame))	
			imsave(outputDir + "output-"+str(fr).replace('.jpg','') + "-rgb" + str(pers) + ".jpg", torch.from_numpy(image).permute(1,2,0).numpy())
			# cv2.imwrite(outputDir + "output-"+str(fr).replace('.jpg','') + "-flow-pers" + str(pers) + ".jpg", visualizeFlow(flow_frame))
			# cv2.imwrite(outputDir + "output-"+str(fr).replace('.jpg','') + "-rgb" + str(pers) + ".jpg", torch.from_numpy(image).permute(1,2,0).numpy()) # convert to brg when using cv2


			### End Loop
			windowIndex = windowIndex + 1
			previousFrame = int(fr.replace(".jpg",''))
			previousIm = fr;

