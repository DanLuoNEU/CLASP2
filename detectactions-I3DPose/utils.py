# Generic imports
import os
import cv2
import time
import math
from math import sqrt
import random
import scipy
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Imports related to PyTorch
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

########## \/ TVNet Util functions \/ ##########

def get_module_list(module, n_modules):
	ml = nn.ModuleList()
	for _ in range(n_modules):
		ml.append(module())
	return ml


def im_tensor_to_numpy(x):
    transpose = transforms.ToPILImage()
    x = np.asarray(transpose(x))
    return x


def save_im_tensor(x, addr):
    x = x.cpu().float()
    transpose = transforms.ToPILImage()
    x = transpose(x[0])
    x.save(addr)


def updateLog(string, file):
    f = open(file, 'a');
    f.write(string + "\n")
    f.close()


def visualizeFlow(of):
    flow = np.zeros((240, 320, 2))
    flow[:, :, 0] = of[0,:,:].cpu().numpy()
    flow[:, :, 1] = of[1,:,:].cpu().numpy()
    hsv = np.zeros((240, 320, 3), dtype=np.uint8)

    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = bgr[...,::-1]

    return rgb

def torch_where(cond, x_1, x_2):
    cond = cond.float()
    return (cond * x_1) + ((1-cond) * x_2)

def meshgrid(height, width, n_repeat):
    # print(height, width, n_repeat)
    x_t = torch.matmul(torch.ones(height, 1),
                        torch.transpose(torch.linspace(-1.0, 1.0, width)[:, None], 1, 0))
    # print(x_t)
    y_t = torch.matmul(torch.linspace(-1.0, 1.0, height)[:, None], torch.ones(1, width))
    # print(y_t)

    x_t_flat = x_t.view(1, -1)
    y_t_flat = y_t.view(1, -1)

    grid = torch.cat([x_t_flat, y_t_flat])[None, ...].cuda().view(-1)
    grid = grid.repeat(n_repeat)
    grid = grid.view(n_repeat, 2, -1)
    grid = grid.permute(0, 2, 1).contiguous().view(n_repeat, height, width, 2)
    # print(grid[0, :, :, 0], x_t)

    return grid

########## ^ TVNet Util functions ^ ##########

def gridRing(N):
    epsilon_low = 0.25
    epsilon_high = 0.15
    rmin = (1-epsilon_low)
    rmax = (1+epsilon_high)
    thetaMin = 0.001
    thetaMax = np.pi/2 - 0.001
    delta = 0.001
    Npole = int(N/4)
    Pool = generateGridPoles(delta, rmin, rmax, thetaMin, thetaMax)
    M = len(Pool)
    idx = random.sample(range(0, M), Npole)
    P = Pool[idx]
    Pall = np.concatenate((P, -P, np.conjugate(P), np.conjugate(-P)), axis=0)

    return P, Pall

# Generate the grid on poles


def generateGridPoles(delta, rmin, rmax, thetaMin, thetaMax):
    rmin2 = pow(rmin, 2)
    rmax2 = pow(rmax, 2)
    xv = np.arange(-rmax, rmax, delta)
    x, y = np.meshgrid(xv, xv, sparse=False)
    mask = np.logical_and(np.logical_and(x**2 + y**2 >= rmin2, x**2 + y ** 2 <= rmax2),
                          np.logical_and(np.angle(x+1j*y) >= thetaMin, np.angle(x+1j*y) <= thetaMax))
    px = x[mask]
    py = y[mask]
    P = px + 1j*py

    return P


'''
# Create Gamma for Fista
def getWeights(Pall,N):
	g2 = pow(abs(Pall),2)
	g2N = np.power(g2,N)

	GNum = 1-g2
	GDen = 1-g2N
	idx = np.where(GNum == 0)[0]

	GNum[idx] = N
	GDen[idx] = pow(N,2)
	G = np.sqrt(GNum/GDen)
	return np.concatenate((np.array([1]),G))


##old way doing Fista
def softshrink(x, lambd):
	# Calculate the masks
	mask1 = x > lambd
	mask2 = x < -lambd

	out = torch.zeros_like(x)
	out += torch.mul( (mask1.float()), (x-lambd))
	out += torch.mul( mask2.float(), (x+lambd))

	return out
def fista(D, Y, Gamma,maxIter,gpu_id):

	DtD = torch.matmul(torch.t(D),D)
	L = torch.norm(DtD,2)
	linv = 1/L
	DtY = torch.matmul(torch.t(D),Y)
	x_old = torch.zeros(DtD.shape[1],DtY.shape[1]).cuda(gpu_id)
	t = 1
	y_old = x_old
	Gamma = torch.mul(Gamma,linv)
	#lambd = lambd*(linv.cpu().numpy())
	A = torch.eye(DtD.shape[1]).cuda(gpu_id) - torch.mul(DtD,linv)

	DtY = torch.mul(DtY,linv)
	lambd = Gamma.view(-1,1).expand(-1,DtY.shape[1])
	for ii in range(maxIter):
		Ay = torch.matmul(A,y_old)
		del y_old
		x_new = softshrink((Ay + DtY),lambd )
		t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
		tt = (t-1)/t_new
		y_old = torch.mul( x_new,(1 + tt))
		y_old -= torch.mul(x_old , tt)
		t = t_new
		x_old = x_new
		del x_new

	#noiseLev = torch.sum(torch.abs(Y - D@x_old))/Y.shape[0]
	noiseLev = torch.sum((Y - D@x_old)**2)/Y.shape[0]
	return x_old,noiseLev
'''


def fista(D, Y, lambd, maxIter):

    DtD = torch.matmul(torch.t(D), D)
    L = torch.norm(DtD, 2)
    linv = 1/L

    # print("D: ", D.shape)
    # print("Dt: ", torch.t(D).shape)
    # print("Y: ", Y.shape)

    DtY = torch.matmul(torch.t(D), Y)
    x_old = Variable(torch.zeros(
        DtD.shape[1], DtY.shape[1]).cuda(), requires_grad=True)
    t = 1
    y_old = x_old
    #lambd = lambd*(linv.cpu().numpy())
    lambd = lambd*(linv.detach().cpu().numpy())
    A = Variable(torch.eye(DtD.shape[1]).cuda(),
                 requires_grad=False) - torch.mul(DtD, linv)

    DtY = torch.mul(DtY, linv)
    Softshrink = nn.Softshrink(lambd)
    with torch.no_grad():
        for ii in range(maxIter):
            Ay = torch.matmul(A, y_old)
            del y_old
            with torch.enable_grad():
                x_new = Softshrink((Ay + DtY))
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
            tt = (t-1)/t_new
            y_old = torch.mul(x_new, (1 + tt))
            y_old -= torch.mul(x_old, tt)
            if torch.norm((x_old - x_new), p=2)/x_old.shape[1] < 1e-4:
                x_old = x_new
                break
            t = t_new
            x_old = x_new
            del x_new

    #noiseLev = torch.sum(torch.abs(Y - D@x_old))/Y.shape[0]
    #noiseLev = torch.sum((Y - D@x_old)**2)/Y.shape[0]

    return x_old


def generateD(rr, theta, row, T):
    W = torch.FloatTensor()
    if isinstance(row, torch.Tensor) and len(row.size()) > 1:
        # print("AHHHH")
        Wones = torch.ones((row.shape[1],1)).cuda()
        Wones = Variable(Wones,requires_grad = False)
        W1 = torch.mul(torch.pow(rr.unsqueeze(0), torch.t(row)),
                       torch.cos(torch.t(row) * theta.unsqueeze(0)))
        W2 = torch.mul(torch.pow(-rr.unsqueeze(0), torch.t(row)),
                       torch.cos(torch.t(row) * theta.unsqueeze(0)))
        W3 = torch.mul(torch.pow(rr.unsqueeze(0), torch.t(row)),
                       torch.sin(torch.t(row) * theta.unsqueeze(0)))
        W4 = torch.mul(torch.pow(-rr.unsqueeze(0), torch.t(row)),
                       torch.sin(torch.t(row) * theta.unsqueeze(0)))
        W = torch.cat((Wones,W1, W2, W3, W4), 1)
        W = W.view(row.shape[1], -1)
    else:
        # print("FUCK")
        Wones = torch.ones((row.shape[1],1)).cuda()
        Wones = Variable(Wones,requires_grad = False)
        W1 = torch.mul(torch.pow(rr, row), torch.cos(row * theta))
        W2 = torch.mul(torch.pow(-rr, row), torch.cos(row * theta))
        W3 = torch.mul(torch.pow(rr, row), torch.sin(row * theta))
        W4 = torch.mul(torch.pow(-rr, row), torch.sin(row * theta))
        W = torch.cat((Wones,W1, W2, W3, W4), 0)
        W = W.view(1, -1)
    return W

# Function to save the checkpoint


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def getListOfFolders(File):
    data = pd.read_csv(File, sep=" ", header=None)[0]
    data = data.str.split('/', expand=True)[1]
    data = data.str.rstrip(".avi").values.tolist()

    return data

# Dataloader for PyTorch.


class videoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, folderList, rootDir, N_FRAME):
        """
        Args:
                N_FRAME (int) : Number of frames to be loaded
                rootDir (string): Directory with all the Frames/Videoes.
                Image Size = 128,160
                2 channels : U and V
        """
        self.listOfFolders = folderList
        self.rootDir = rootDir
        self.nfra = N_FRAME
        self.x_fra = 240
        self.y_fra = 320
        self.numpixels = self.x_fra * self.y_fra

    def __len__(self):
        return len(self.listOfFolders)

    def readData(self, folderName):
        path = os.path.join(self.rootDir, folderName)
        OF = torch.FloatTensor(2, self.nfra, self.numpixels)
        OFori = torch.FloatTensor(1, 2, self.nfra, self.x_fra, self.y_fra)
        for framenum in range(self.nfra):
            flow = np.load(os.path.join(path, str(framenum)+'.npy'))
            flow = np.transpose(flow, (2, 0, 1))
            OFori[:, :, framenum, :, :] = torch.from_numpy(
                flow).type(torch.FloatTensor).unsqueeze(0)
        # flows = alignOF(OFori, self.nfra-1)
        # if random.randint(1, 50) == 1:
        #     showOFs(flows, OFori, self.nfra, 'of')
        # OF = flows.view((2, self.nfra, self.numpixels)).type(torch.FloatTensor)
        OFori = OFori.view((2, self.nfra, self.numpixels)
                           ).type(torch.FloatTensor)
        return OFori

    def __getitem__(self, idx):
        folderName = self.listOfFolders[idx]
        Frame = self.readData(folderName)
        sample = {'frames': Frame}
        return sample


def warp(input, tensorFlow):

    torchHorizontal = torch.linspace(-1.0, 1.0, input.size(3))
    torchHorizontal = torchHorizontal.view(1, 1, 1, input.size(3)).expand(input.size(0), 1, input.size(2), input.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, input.size(2))
    torchVertical = torchVertical.view(1, 1, input.size(2), 1).expand(input.size(0), 1, input.size(2), input.size(3))

    tensorGrid = torch.cat([torchHorizontal, torchVertical], 1).cuda()
    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((input.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((input.size(2) - 1.0) / 2.0)], 1)
    # tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / 10, tensorFlow[:, 1:2, :, :] / 20], 1)

    # print(torch.mean(input-tensorFlow))
    return torch.nn.functional.grid_sample(input=input.cuda(), grid=(tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
