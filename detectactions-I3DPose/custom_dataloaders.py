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

class clasp_sliding_window_frameset(data.Dataset):
	def __init__(self, folderList, rootDir, N_FRAME):
		self.folderList = folderList
		self.rootDir = rootDir
		self.nfra = N_FRAME
		self.numpixels = 320 * 240 # 640 * 480 # 1920*1080
		self.videoFrameProg = 1

	def readRGBData(self, folderName):
		path = os.path.join(self.rootDir,folderName)
		
		print(path)

		bunch_of_frames = []

		for index in range(0, self.nfra):
			frames = torch.FloatTensor(3,self.nfra,self.numpixels)

			end = len([name for name in sorted(os.listdir(path)) if ".jpg" in name])
			offset = [name for name in sorted(os.listdir(path)) if ".jpg" in name]
			# print(offset[0])			
			# print("offset: ", str(int(offset[0].replace('.jpg',''))))
			offset = int(offset[0].replace('.jpg',''))
			# input()

			offset = offset + self.videoFrameProg # random.randint(offset, (end - self.nfra) - 2)
			# offset = 1;

			### NEED N + 1 frames when starting with raw frames
			frames = torch.zeros(3, self.nfra + 1, 240, 320)
			i = 0
			for framenum in range(offset, (self.nfra) + offset):
				print("reading from frame " + str('%04d' % framenum) + "...")
				# print("For rfolder: ", folderName)
				img_path1 = os.path.join(path, str('%07d' % framenum) + '.jpg')		
				image1 = Image.open(img_path1)
				image1 = ToTensor()(image1)
				image1 = image1.float()				
				# print(image1.shape)
				data = np.transpose(image1.numpy(), (1, 2, 0)) # put height and width in front
				data = skimage.transform.resize(data, (240, 320)) 
				image1 = torch.from_numpy(np.transpose(data, (2, 0, 1) ) ) # move back 
				# print(image1.shape)
				frames[:,i,:] = image1 # .view(3, 480 * 640)	
				i = i + 1

			bunch_of_frames = bunch_of_frames + [frames]
		
		return bunch_of_frames

	def __len__(self):
		return len(self.folderList)

	def __getitem__(self, index):
		folderName = self.folderList[index]
		bunch_of_frames = []

		bunch_of_frames = self.readRGBData(folderName)
		# sample = { 'frames': Frame , 'ac' : ac}
		self.videoFrameProg = self.videoFrameProg + 1;
		print("self.videoFrameProg: ", self.videoFrameProg)
		return bunch_of_frames

class clasp_frameset(data.Dataset):
	def __init__(self, folderList, rootDir, N_FRAME):
		self.folderList = folderList;
		self.rootDir = rootDir;
		self.nfra = N_FRAME;
		self.perVideo = 5;
		self.numpixels = 320 * 240 # 640 * 480 # 1920*1080

	def readRGBData(self, folderName):
		path = os.path.join(self.rootDir,folderName)
		
		print(path)

		bunch_of_frames = []

		for index in range(0, self.perVideo):
			frames = torch.FloatTensor(3,self.nfra,self.numpixels)

			end = len([name for name in sorted(os.listdir(path)) if ".jpg" in name])
			offset = [name for name in sorted(os.listdir(path)) if ".jpg" in name];
			# print(offset[0])			
			# print("offset: ", str(int(offset[0].replace('.jpg',''))))
			offset = int(offset[0].replace('.jpg',''))
			# input()

			offset = random.randint(offset, (end - self.nfra) - 2)
			# offset = 1;

			### NEED N + 1 frames when starting with raw frames
			frames = torch.zeros(3, self.nfra + 1, 240, 320);
			i = 0;
			for framenum in range(offset, 2 * (self.nfra) + offset, 2):
				# print("reading from frame " + str('%04d' % framenum) + "...")
				# print("For rfolder: ", folderName)
				img_path1 = os.path.join(path, str('%07d' % framenum) + '.jpg')		
				image1 = Image.open(img_path1)
				image1 = ToTensor()(image1)
				image1 = image1.float()				
				# print(image1.shape)
				data = np.transpose(image1.numpy(), (1, 2, 0)) # put height and width in front
				data = skimage.transform.resize(data, (240, 320)) 
				image1 = torch.from_numpy(np.transpose(data, (2, 0, 1) ) ) # move back 
				# print(image1.shape)
				frames[:,i,:] = image1 # .view(3, 480 * 640)	
				i = i + 1;

			bunch_of_frames = bunch_of_frames + [frames]
		
		return bunch_of_frames

	def __len__(self):
		return len(self.folderList)

	def __getitem__(self, index):
		folderName = self.folderList[index]
		bunch_of_frames = []

		bunch_of_frames = self.readRGBData(folderName)
		# sample = { 'frames': Frame , 'ac' : ac}

		return bunch_of_frames
		


class frame_dataset(data.Dataset):
    def __init__(self, args):
        self.frame_dir = args.frame_dir
        
        self.frame_addr = np.asarray([os.path.join(self.frame_dir, addr) for addr in os.listdir(self.frame_dir)])
        self.frame_addr.sort()
        self.to_tensor = get_transfrom()
        self.img_size = Image.open(self.frame_addr[0]).convert('RGB').size[::-1]
    
    def __len__(self):
        return self.frame_addr.shape[0] - 1

    def __getitem__(self, index):
        frame_1 = self.to_tensor(Image.open(self.frame_addr[index]).convert('RGB')).float().cuda()
        frame_2 = self.to_tensor(Image.open(self.frame_addr[index+1]).convert('RGB')).float().cuda()
        return frame_1, frame_2

## Dataloader for PyTorch.
class videoDatasetTVNETOF(Dataset):
	"""Dataset Class for Loading Video"""

	def __init__(self, folderList, rootDir, N_FRAME):

		"""
		Args:
			N_FRAME (int) : Number of frames to be loaded
			rootDir (string): Directory with all the Frames/Videoes.
			Image Size = 240,320
			2 channels : U and V
		"""

		self.listOfFolders = folderList
		self.rootDir = rootDir
		self.nfra = N_FRAME
		self.numpixels = 240*320 # If Kitti dataset, self.numpixels = 128*160
		self.action_classes = {}
		self.action_classes = self.actionClassDictionary();
		print("Loaded " + str(len(self.action_classes)) + " action classes...")

	def __len__(self):
		return len(self.listOfFolders)

	def getACDic(self):
		return self.action_classes;

	def actionClassDictionary(self):
		num_classes = 0;
		for folderName in sorted(self.listOfFolders):
			result = re.search('v_(.*)_g', folderName)
			n = result.group(1)
			if n in self.action_classes.keys():
				continue;
			else:
				# print(str(n) + " -- " + str(num_classes))
				self.action_classes[n] = num_classes;
				num_classes = num_classes + 1;

		return self.action_classes

	def readData(self, folderName):
		path = os.path.join(self.rootDir,folderName)
		OF = torch.FloatTensor(2,self.nfra,self.numpixels)
		for framenum in range(self.nfra):
			flow = np.load(os.path.join(path,str(framenum)+'.npy'))
			# print("Reading ", os.path.join(path,str(framenum)+'.npy'), "...")
			# print("Flow: ", flow.shape)
			# flow = np.transpose(flow,(2,0,1))
			OF[:,framenum] = torch.from_numpy(flow.reshape(2,self.numpixels)).type(torch.FloatTensor)
		
		return OF

	def __getitem__(self, idx):
		folderName = self.listOfFolders[idx]

		result = re.search('v_(.*)_g', folderName)
		n = result.group(1)
		if n in self.action_classes.keys():
			ac = self.action_classes[n]
		else:
			input("Found new action class???")
			self.action_classes[n] = self.num_classes
			self.num_classes = self.num_classes + 1;
			ac = self.action_classes[n]

		Frame = self.readData(folderName)
		sample = { 'frames': Frame , 'ac' : ac }

		return sample

## Dataloader for PyTorch.
class videoDatasetRandomRawFrames(Dataset):
	"""Dataset Class for Loading Video"""

	def __init__(self, folderList, rootDir, N_FRAME):

		"""
		Args:
			N_FRAME (int) : Number of frames to be loaded
			rootDir (string): Directory with all the Frames/Videoes.
			Image Size = 240,320
			2 channels : U and V
		"""

		self.listOfFolders = folderList
		self.rootDir = rootDir
		self.nfra = N_FRAME
		self.img_size = (240, 320)
		self.numpixels = 240*320 # If Kitti dataset, self.numpixels = 128*160
		self.action_classes = {}
		self.action_classes = self.actionClassDictionary();
		print("Loaded " + str(len(self.action_classes)) + " action classes...")

	def __len__(self):
		return len(self.listOfFolders)

	def getACDic(self):
		return self.action_classes;

	def actionClassDictionary(self):
		num_classes = 0;
		for folderName in sorted(self.listOfFolders):
			result = re.search('v_(.*)_g', folderName)
			n = result.group(1)
			if n in self.action_classes.keys():
				continue;
			else:
				# print(str(n) + " -- " + str(num_classes))
				self.action_classes[n] = num_classes;
				num_classes = num_classes + 1;

		return self.action_classes

	def readRGBData(self, folderName):
		# path = os.path.join(self.rootDir,folderName)
		# img = torch.FloatTensor(3, self.nfra,self.numpixels)
		path = os.path.join(self.rootDir,folderName)
		frames = torch.FloatTensor(3,self.nfra,self.numpixels)

		offset = len([name for name in sorted(os.listdir(path)) if ".jpeg" in name]);
		offset = random.randint(1, offset - self.nfra - 1)

		# print("reading " + str(self.nfra)  + " data")

		### NEED N + 1 frames when starting with raw frames
		frames = torch.zeros(3, self.nfra, 240, 320);
		# frames = torch.zeros(self.nfra, 3, 240, 320)
		for framenum in range(offset, self.nfra + offset):
			
			# print("reading from frame " + str(framenum) + "...")
			img_path1 = os.path.join(path, "image-" + str('%04d' % framenum) + '.jpeg')		
			image1 = Image.open(img_path1)
			image1 = ToTensor()(image1)
			image1 = image1.float()		
			# print(image1.shape)
			# image1 = image1.view(-1, 240*320)
			# print(image1.shape)			
			
			frames[:,framenum - offset,:] = image1
		# print(frames.shape)	
		return frames

	def readOFData(self, folderName):
		path = os.path.join(self.rootDir,folderName)
		OF = torch.FloatTensor(2,self.nfra,self.numpixels)
		for framenum in range(self.nfra):
			flow = np.load(os.path.join(path,str(framenum)+'.npy'))
			flow = np.transpose(flow,(2,0,1))
			OF[:,framenum] = torch.from_numpy(flow.reshape(2,self.numpixels)).type(torch.FloatTensor)
		
		return OF

	def __getitem__(self, idx):
		folderName = self.listOfFolders[idx]

		result = re.search('v_(.*)_g', folderName)
		n = result.group(1)
		if n in self.action_classes.keys():
			ac = self.action_classes[n]
		else:
			input("Found new action class???")
			self.action_classes[n] = self.num_classes
			self.num_classes = self.num_classes + 1;
			ac = self.action_classes[n]

		Frame = self.readRGBData(folderName)
		sample = { 'frames': Frame , 'ac' : ac }

		return sample

## Dataloader for PyTorch.
class videoDatasetStaticRawFrames(Dataset):
	"""Dataset Class for Loading Video"""

	def __init__(self, folderList, rootDir, flowDir, N_FRAME):

		"""
		Args:
			N_FRAME (int) : Number of frames to be loaded
			rootDir (string): Directory with all the Frames/Videoes.
			Image Size = 240,320
			2 channels : U and V
		"""

		self.listOfFolders = folderList
		self.rootDir = rootDir
		self.flowDir = flowDir
		self.nfra = N_FRAME
		self.img_size = (240, 320)
		self.numpixels = 240*320 # If Kitti dataset, self.numpixels = 128*160
		self.action_classes = {}
		self.action_classes = self.actionClassDictionary();
		print("Loaded " + str(len(self.action_classes)) + " action classes...")

	def __len__(self):
		return len(self.listOfFolders)

	def getACDic(self):
		return self.action_classes;

	def actionClassDictionary(self):
		num_classes = 0;
		for folderName in sorted(self.listOfFolders):
			result = re.search('v_(.*)_g', folderName)
			n = result.group(1)
			if n in self.action_classes.keys():
				continue;
			else:
				# print(str(n) + " -- " + str(num_classes))
				self.action_classes[n] = num_classes;
				num_classes = num_classes + 1;

		return self.action_classes

	def readRGBData(self, folderName):
		# path = os.path.join(self.rootDir,folderName)
		# img = torch.FloatTensor(3, self.nfra,self.numpixels)
		path = os.path.join(self.rootDir,folderName)
		frames = torch.FloatTensor(3,self.nfra,self.numpixels)

		# offset = len([name for name in sorted(os.listdir(path)) if ".jpeg" in name]);
		# offset = random.randint(1, offset - self.nfra - 1)
		offset = 1;
		# print("reading " + str(self.nfra)  + " data")

		### NEED N + 1 frames when starting with raw frames
		frames = torch.zeros(3, self.nfra + 1, 240, 320);
		i = 0;
		# frames = torch.zeros(self.nfra, 3, 240, 320)
		for framenum in range(offset, 2 * (self.nfra + offset) ,2):
			# print("reading from frame " + str('%04d' % framenum) + "...")
			# print("For rfolder: ", folderName)
			img_path1 = os.path.join(path, "image-" + str('%04d' % framenum) + '.jpeg')		
			image1 = Image.open(img_path1)
			image1 = ToTensor()(image1)
			image1 = image1.float()		
			# print(image1.shape)
			# image1 = image1.view(-1, 240*320)
			# print(image1.shape)			
			
			frames[:,i,:] = image1
			i = i + 1;
		# print(frames.shape)	
		return frames

	def readOFData(self, folderName):
		path = os.path.join(self.flowDir,folderName)
		OF = torch.FloatTensor(2,self.nfra,self.numpixels)
		i = 0;
		for framenum in range(0, 2 * self.nfra, 2):
			# print("reading in OF: " + str('%04d' % framenum) + ".npy ...")
			flow = np.load(os.path.join(path,str(framenum)+'.npy'))
			# flow = np.transpose(flow,(2,0,1))
			OF[:,i] = torch.from_numpy(flow.reshape(2,self.numpixels)).type(torch.FloatTensor)
			i = i + 1;

		return OF

	def __getitem__(self, idx):
		folderName = self.listOfFolders[idx]

		result = re.search('v_(.*)_g', folderName)
		n = result.group(1)
		if n in self.action_classes.keys():
			ac = self.action_classes[n]
		else:
			input("Found new action class???")
			self.action_classes[n] = self.num_classes
			self.num_classes = self.num_classes + 1;
			ac = self.action_classes[n]

		Frame = self.readRGBData(folderName)
		Flows = self.readOFData(folderName)
		sample = { 'frames': Frame , 'ac' : ac, 'flows' : Flows }

		return sample

