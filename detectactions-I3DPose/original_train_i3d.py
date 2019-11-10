import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import numpy as np
from pytorch_i3d import InceptionI3d
from charades_dataset import Charades as Dataset

from tvnet1130_train_options import arguments
from model.network import model as tvnet

# Random Seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

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


### TVNet
tvnet_args = arguments().parse() # tvnet parameters
tvnet_args.batch_size = 4; # 3 + 1 for UCF101
tvnet_args.data_size = [tvnet_args.batch_size, 3, 240, 320]
# init_file = '/home/truppr/kDYAN-TV/savedModels/UCF-TVNet-1-1-30.pth'
init_file = '/home/truppr/kDYAN-TV/savedModels/UCF-DYAN-TVNET-1-1-30-E2E-DL_1_0.3163.pth'
of = tvnet(tvnet_args).cuda(0)

of = loadTVNet(of, init_file)
of = of

of = nn.DataParallel(of)


def run(init_lr=0.1, max_steps=64e3, mode='rgb', root='/storage/truppr/CHARADES/Charades_v1_rgb', train_split='charades/charades.json', batch_size=16, save_model=''):
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    # print(root)
    print("creating training set...")
    dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=18, pin_memory=True)

    print("creating validation set...")
    val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=18, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    print("setting up the model...")
    if mode == 'flow' or mode == 'rgb':
        if mode == 'flow':
            i3d = InceptionI3d(400, in_channels=2)
            i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
        elif mode == 'rgb':
            i3d = InceptionI3d(400, in_channels=3)
            i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
        i3d.replace_logits(157) # number of classes... originally 157
        i3d.cuda(0)
        i3d = nn.DataParallel(i3d)

    elif mode == 'both':
        i3d_rgb = InceptionI3d(400, in_channels=3)
        i3d_rgb.load_state_dict(torch.load('models/rgb_imagenet.pt'))

        i3d_flow = InceptionI3d(400, in_channels=2)
        i3d_flow.load_state_dict(torch.load('models/flow_imagenet.pt'))
        
        i3d_rgb.replace_logits(157) # number of classes... originally 157
        i3d_flow.replace_logits(157)

        i3d_rgb.cuda(0)
        i3d_flow.cuda(0)

        i3d_rgb = nn.DataParallel(i3d_rgb)
        i3d_flow = nn.DataParallel(i3d_flow)


    lr = init_lr
    
    if mode == 'both':
        optimizer_rgb = optim.SGD(i3d_rgb.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
        optimizer_flow = optim.SGD(i3d_flow.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
        lr_sched_rgb = optim.lr_scheduler.MultiStepLR(optimizer_rgb, [300, 1000])
        lr_sched_flow = optim.lr_scheduler.MultiStepLR(optimizer_flow, [300, 1000])
    else:
        optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
        lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    num_steps_per_update = 4 # accum gradient
    steps = 0
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        # print 'Step {}/{}'.format(steps, max_steps)
        # print '-' * 10
        print('Step ' + str(steps) + '/' + str(max_steps))
        print('-' * 25)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                print("training model...")
                if mode == 'both':
                    i3d_rgb.train(True)
                    i3d_flow.train(True) 
                    optimizer_rgb.zero_grad()
                    optimizer_flow.zero_grad()
                else:
                    i3d.train(True)
                    optimizer.zero_grad()
            else:
                print("validating model...")
                if mode == 'both':
                    i3d_rgb.train(False)
                    i3d_flow.train(False)
                    optimizer_rgb.zero_grad()
                    optimizer_flow.zero_grad()
                else:
                    i3d.train(False)  # Set model to evaluate mode
                    optimizer.zero_grad()
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            # optimizer.zero_grad()
            print("zeroed...")
            # print(len(dataloaders["train"]))
            # print(dataloaders["train"])
            # Iterate over data.
            for data in dataloaders[phase]:
                # print("starting iter...")
                
                num_iter += 1
                # get the inputs
                inputs, labels = data

                print("data size: ", inputs.shape, " label: ", labels)

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                torch.set_printoptions(profile="full")
                print("labels:\n", labels)
                print("labels:\n", labels.shape)
                print("Inputs: \n", inputs.shape)
                torch.set_printoptions(profile="default")

                if mode == 'both':
                    per_frame_logits = i3d_rgb(inputs)
                    per_flows_logits = i3d_flow(flow_inputs)
                else:
                    per_frame_logits = i3d(inputs)

                    # upsample to input size
                    per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.item()

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
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
                        # print '{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10)
                        print(str(phase) + ' Loc Loss: ' + str(tot_loc_loss/(10*num_steps_per_update)) + ' Cls Loss: ' + str(tot_cls_loss/(10*num_steps_per_update)) + ' Tot Loss: ' + str(tot_loss/10))
                        # save model
                        torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'-'+str(tot_loss/10)+'.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
                #else:
                #    print(str(phase) + ' Loc Loss: ' + str(tot_loc_loss/(10*num_steps_per_update)) + ' Cls Loss: ' + str(tot_cls_loss/(10*num_steps_per_update)) + ' Tot Loss: ' + str(tot_loss/10))
                
            if phase == 'val':
                print(str(phase) + ' Loc Loss: ' + str(tot_loc_loss/num_iter).zfill(4) + ' Cls Loss: ' + str( tot_cls_loss/num_iter).zfill(4) + ' Tot Loss: ' + str((tot_loss*num_steps_per_update)/num_iter).zfill(4))
                # print '{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter) 
            print("whoops...")


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, save_model=args.save_model)
