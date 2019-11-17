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
sys.path.insert(0, './model')
from network import model as tvnet
tvnet_args = arguments().parse() # tvnet parameters
tvnet_args.batch_size = 4; # 3 + 1 for UCF101
tvnet_args.data_size = [tvnet_args.batch_size, 3, 240, 320]
# init_file = '/home/truppr/kDYAN-TV/savedModels/UCF-TVNet-1-1-30.pth'
init_file = '/home/truppr/kDYAN-TV/savedModels/UCF-DYAN-TVNET-1-1-30-E2E-DL_1_0.3163.pth'
of = tvnet(tvnet_args).cuda()

of = loadTVNet(of, init_file)

of = of.cpu()


class ava_dataset(data.Dataset):
    def __init__(self, folderList, frameDir, flowDir, jsonDir):
        self.folderList = folderList;
        self.frameDir = frameDir;
        self.jsonDir = jsonDir;
        self.flowDir = flowDir;
        # self.numpixels = 320 * 240 # 640 * 480 # 1920*1080
        # self.videoFrameProg = 1;

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
        
        # print(height, width)
        return height, width

    def within(self, keypoints, tup):
        boolean = False
        score = 0

        for item in keypoints:
            if tup[0] <= item[0] <= tup[1] and  tup[2] <= item[1] <= tup[3]:
                boolean = True;
                score = score + 1

        # print(keypoints)
        # print(tup)

        return boolean, score

    # (altered but thank you): https://github.com/leaderj1001/Action-Localization/tree/master/video_crawler 
    def prepare(self, sample, video_path, jsonDir):
        tube = {}
        video_name = sample[0]
        start_time = int(sample[1])
        start_time = int(start_time) - 1
        end_time = int(start_time) + 3
        # print(sample)
        # print("Searching ", video_path, "for", video_name)
        

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
            # print(command)

            try:
                print("Processing video: {}".format(video_name))
                output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                print('status :: ', status, ', error print :: ', err.output.decode('euc-kr'))
                return status, err.output

            # if os.path.exists(cropped_video_filename)
            
            os.mkdir(cropped_video_filename.split(".")[0])
            os.mkdir(cropped_video_filename.split(".")[0] + "/scene/")
            os.mkdir(cropped_video_filename.split(".")[0] + "/rgb/")
            os.mkdir(cropped_video_filename.split(".")[0] + "/joint/")
            command = "ffmpeg -i " + cropped_video_filename + " -vf \"select=not(mod(n\,1))\" -vsync vfr -q:v 2 "  + cropped_video_filename.split(".")[0] + "/scene/%07d.jpg"
            # print(command)

            try:
                print("Splicing video: {}".format(video_name))
                output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                print('status :: ', status, ', error print :: ', err.output.decode('euc-kr'))
                return status, err.output
            
            # Extract JOINT info
            # img_directory = cropped_video_filename.split(".")[0] + "/scene/" 
            # command = "cd ./AlphaPose/ && /home/truppr/anaconda3/envs/dyanEnv/bin/python demo.py --indir " + img_directory + " --outdir " + cropped_video_filename.split(".")[0] + "/joint/" + " --fast_inference False"
            command = "cd ./AlphaPose/ && /home/truppr/anaconda3/envs/dyanEnv/bin/python video_demo.py --video " + cropped_video_filename + " --outdir " + cropped_video_filename.split(".")[0] + "/joint/" + " --save_video --vis_fast"
            print(command)

            try:
                print("Capturing Pose INFO: {}".format(video_name))
                output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                print('status :: ', status, ', error print :: ', err.output.decode('euc-kr'))
                return status, err.output

            with open(cropped_video_filename.split(".")[0] + "/joint/" + 'alphapose-results.json') as json_file:
                data = json.load(json_file)
            
            # print("extracting data...")
            i = 0;
            new_data = {}
            for entry in data:
                try:
                    # print("try")
                    new_data[str(data[i]["category_id"])]
                except KeyError:
                    # print("caught")
                    new_data[str(data[i]["category_id"])] = {}
                '''
                try:
                    print("try")
                    new_data[str(data[i]["category_id"])][data[i]["image_id"]]
                except KeyError:
                    print("caught")
                    new_data[str(data[i]["category_id"])][data[i]["image_id"]] 
                '''
                f = str(int(data[i]["image_id"].split(".")[0]) + 1).zfill(7) + "." + str(data[i]["image_id"].split(".")[1])
                new_data[str(data[i]["category_id"])][f] = []
                for eh in range(0, len(data[i]["keypoints"]), 3):
                    new_data[str(data[i]["category_id"])][f].append((data[i]["keypoints"][eh], data[i]["keypoints"][eh + 1], data[i]["keypoints"][eh + 2]))  
                
                i = i + 1

            print("extracting relevent data...")
            friend = -1 # this is the person who we are interested in...
            friend_score = {}
            for person in new_data:
                print(str(person))
                for frame in new_data[person]:
                    res, num = self.within(new_data[person][frame], (int(float(sample[3]) * height), int(float(sample[5]) * height), int(float(sample[2]) * width), int(float(sample[4]) * width)))
                    if res:
                        try:
                            # print("try")
                            friend_score[str(person)]
                        except KeyError:
                            # print("caught")
                            friend_score[str(person)] = 0
                        friend = person
                        friend_score[person] = friend_score[person] + num

            for peep in new_data:
                print(peep)

            #input()

            if friend == -1:
                pass # print("Uh oh...")
            else:
                # print("is this new data?")
                # print("person ", str(max(friend_score.iteritems(), key=operator.itemgetter(1))[0]))
                peep = str(max(friend_score.iteritems(), key=operator.itemgetter(1))[0])
                new_data = new_data[peep]

            # print(new_data)

            # for peep in new_data:
             #    print(peep)

            with open(cropped_video_filename.split(".")[0] + "/joint/" + 'results.json', 'w') as fp:
                json.dump(new_data, fp)
            # input()
        '''
        rgb_stream = torch.zeros(1, 3, 90, 224, 224) # torch.zeros(1,3,64,224,224)
        onlyfiles = [f for f in listdir(cropped_video_filename.split(".")[0] + "/scene/") if isfile(join(cropped_video_filename.split(".")[0] + "/scene/", f))]
        index = 0;
        for f in sorted(onlyfiles):
            i = imread(join(cropped_video_filename.split(".")[0] + "/scene/", f))

            # Visualize bounding boxes...
            #i = cv2.rectangle(i, (int(float(sample[2]) * width), int(float(sample[3]) * height)), (int(float(sample[4]) * width), int(float(sample[5]) * height)), (255, 255, 0), 2)
            #if int(f.split(".")[0]) > 30 and int(f.split(".")[0]) < 60:
            #    i = cv2.putText(i, "Person " + sample[7], (int(float(sample[2]) * width) + 10, int(float(sample[3]) * height) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
            #    i = cv2.putText(i, "Action " + sample[6], (int(float(sample[2]) * width) + 10, int(float(sample[3]) * height) + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
            

            # Save RGB Stream Input

            i = i[int(float(sample[3]) * height):int(float(sample[5]) * height),int(float(sample[2]) * width):int(float(sample[4]) * width),:]
            i = skimage.transform.resize(i, (224,224, 3))
            imsave(join(cropped_video_filename.split(".")[0] + "/rgb/", f), i)
            i = np.transpose(i, (2, 0, 1))

            rgb_stream[0,:,index,:,:] = torch.from_numpy(i)
            index -= -1;

        '''
        # return rgb_stream
        # print(cropped_video_filename.split(".")[0] + "/joint/" + 'results.json')
        with open(cropped_video_filename.split(".")[0] + "/joint/" + 'results.json', 'r') as json_file:
            new_data = json.load(json_file)

        # print(cropped_video_filename.split(".")[0], new_data, height, width)
        return cropped_video_filename.split(".")[0], new_data, height, width

    def viableActionTube(self, imgs):
        previous = int(imgs[0].replace(".jpg",""))
        count = 0
        broken = 0
        chunks = []

        # print(imgs)
        # input()

        for im in imgs[1:]:
            if int(im.replace(".jpg",'')) - previous <= 5:
                count = count + 1
            else:
                if count >= 64:
                    return True, imgs[broken:]
                
                chunks.append(imgs[broken:count])
                
                broken = count
                count = 0
            previous = int(im.replace(".jpg",''))

        if count >= 64:
            return True, imgs[broken:]
        else:
            l = 0
            curr = []
            for chunk in chunks:
                if len(chunk) > l:
                    curr = chunk
                    l = len(chunk)
            print("best len:", l)
            return False, curr

    def extractTube(self, path, sample, height, width, joints):
        dublicate = False
        rgb_stream = torch.zeros(1, 3, 64, 224, 224)
        scene_stream = torch.zeros(1, 3, 64, height, width)
        delete_keys = []

        res, keys = self.viableActionTube(sorted(joints.keys()))

        if res:
            if len(keys) >= 65:
                start = len(keys) - 64
                delete_keys = sorted(joints.keys())[0:start]
            elif len(keys) <= 20:
                return "ERR"
        elif not res:
            if len(keys) <= 20:
                return "ERR"
        else:
            p = (64 - len(keys)) / 64.0
            dublicate = True

        rgb_index = 0
        # for item in sorted(keys):
        # print("some keys for ya:", sorted(joints.keys()))
        bb = []
        ehpi = np.zeros((64, 17, 3))
        for item in sorted(joints.keys()):
            # print(joints[item])

            x_arr = []
            y_arr = []
            xmin = 0
            xmax = 0
            ymin = 0
            ymax = 0
            # print(item)
            # truncate beginning if too long
            if item in delete_keys:
                del joints[item]
                continue

            keypoints = joints[item]
            label = ''

            for key in keypoints:
                x_arr.append(key[0])
                y_arr.append(key[1])
            
            label = "using joints for BB"
            # print("using joints for BB!!!")
            buffer = 75
            xmin = int(min(x_arr)) - buffer 
            xmax = int(max(x_arr)) + buffer 
            ymin = int(min(y_arr)) - buffer
            ymax = int(max(y_arr)) + buffer
            '''
            if score / len(x_arr) > threshold:
                label = "using joints for BB"
                # print("using joints for BB!!!")
                buffer = 10
                xmin = int(min(x_arr)) - buffer 
                xmax = int(max(x_arr)) + buffer 
                ymin = int(min(y_arr)) - buffer
                ymax = int(max(y_arr)) + buffer
            else:
                label = "using GT for BB"
                # print("Using GT for BB!!!")
                buffer = 10
                xmin = int(float(sample[2]) * width) - buffer 
                xmax = int(float(sample[4]) * width) + buffer 
                ymin = int(float(sample[3]) * height) - buffer
                ymax = int(float(sample[5]) * height) + buffer
            '''

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > width:
                xmax = width
            if ymax > height:
                ymax = height

            # ENCODE HUMAN POSE IMAGE
            # print(len(keypoints))
            k = 0
            for key in keypoints:
                ehpi[rgb_index, k, 0] = 0
                ehpi[rgb_index, k, 1] = (key[0] - xmin) / (xmax - xmin)
                ehpi[rgb_index, k, 2] = (key[1] - ymin) / (ymax - ymin)
                k = k + 1

            # print(item)
            # print(join(path + "/scene/", str(int(item.split(".")[0])).zfill(7)+ "." + item.split(".")[1]))
            i = imread(join(path + "/scene/", str(int(item.split(".")[0])).zfill(7) + "." + item.split(".")[1]))
            # print(i.shape)

            # Visualize bounding boxes in scene...
            # print(int(float(sample[2]) * width), int(float(sample[3]) * height), int(float(sample[4]) * width), int(float(sample[5]) * height))
            # scene = cv2.rectangle(i, (int(float(sample[2]) * width), int(float(sample[3]) * height)), (int(float(sample[4]) * width), int(float(sample[5]) * height)), (255, 255, 0), 2)
            # scene = cv2.putText(i, "Person " + sample[7], (int(float(sample[2]) * width) + 10, int(float(sample[3]) * height) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
            # scene = cv2.putText(i, "Action " + sample[6], (int(float(sample[2]) * width) + 10, int(float(sample[3]) * height) + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
            scene = cv2.rectangle(i, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
            scene = cv2.putText(i, "Person " + sample[7], (xmin + 10, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
            scene = cv2.putText(i, "Action " + sample[6], (xmin + 10, ymin + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
            scene = cv2.putText(i, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

            imsave(join(path + "/scene/", item), scene)
            bb.append(((xmin, ymin, xmax, ymax)))

            # Visualize rgb stream...
            # print(xmin,xmax,ymin,ymax)
            # print(i.shape)
            
            original_image = np.transpose(i, (2, 0, 1)) # i;
            i = i[ymin:ymax,xmin:xmax,:]
            i = skimage.transform.resize(i, (224,224, 3))
            # print("hey")
            # print(join(path + "/rgb/", item))
            # print(i.shape)
            imsave(join(path + "/rgb/", item), i)
            # print("bby", rgb_index)
            i = np.transpose(i, (2, 0, 1))
            rgb_stream[0, :, rgb_index, :, :] = torch.from_numpy(i)
            scene_stream[0, :, rgb_index, :, :] = torch.from_numpy(original_image)
            
            # print("yooo")
            if dublicate and random.uniform(0, 1) < p:
                try:
                    # print("work...")
                    rgb_stream[0, :, rgb_index + 1, :, :] = torch.from_numpy(i)
                    scene_stream[0, :, rgb_index + 1, :, :] = torch.from_numpy(original_image)
                    ehpi[rgb_index + 1, :, :] = ehpi[rgb_index, :, :]
                    rgb_index = rgb_index + 1
                except IndexError:
                    print("overpadded... continuing...")
                    break
                except ValueError:
                    print(rgb_index)

            rgb_index = rgb_index + 1
            if rgb_index > 63:
                break
        # print("i dont understand")
        # copy last frame if too short

        if rgb_index < 63:
            # print(rgb_index)
            for i in range(rgb_index + 1, 63):
                # print("peep", i)
                rgb_stream[0, :, i, :, :] = rgb_stream[0, :, rgb_index, :, :]
                scene_stream[0, :, i, :, :] = scene_stream[0, :, rgb_index, :, :]
                ehpi[i, :, :] = ehpi[rgb_index, :, :]

        # print("return")
        return rgb_stream, scene_stream, ehpi, bb

    def extractFlow(self, path, tube, h, w, bb):
        flow_frame = torch.zeros((64, 2, 224, 224))
        path = path + "/flows/"
        os.mkdir(path)
        for index in range(1, 64):
            # x1 = frame_to_tensor(image_path+previousIm, hf).type(torch.FloatTensor) # .cuda();
            # x2 = frame_to_tensor(image_path+fr).type(torch.FloatTensor) # .cuda();
            # image = x2
            
            data = skimage.transform.resize(tube[0,:,index - 1, :, :].cpu().numpy(), (3, 240, 320))
            # data = np.transpose(data, (1, 2, 0))
            x1 = torch.from_numpy(data)
            # print("x1:", x1.shape)
            data = skimage.transform.resize(tube[0,:,index, :, :].cpu().numpy(), (3, 240, 320))
            # data = np.transpose(data, (1, 2, 0))
            x2 = torch.from_numpy(data)
            # print("x2:", x2.shape)
            u1, u2 = of(x2.unsqueeze(0), x1.unsqueeze(0), need_result=True)

            # print("u1:", u1.shape)
            data = u1.detach()[0, 0, bb[index][1]:bb[index][3], bb[index][0]:bb[index][2]]
            data = skimage.transform.resize(data.cpu().numpy(), (1, 1, 224, 224))
            flow_frame[index, 0,:,:] = torch.from_numpy(data)

            data = u2.detach()[0, 0, bb[index][1]:bb[index][3], bb[index][0]:bb[index][2]]
            data = skimage.transform.resize(data.cpu().numpy(), (1,1, 224, 224))
            flow_frame[index, 1,:,:] = torch.from_numpy(data)

        np.save(path + "flows.npy", flow_frame.numpy())

        return flow_frame


    def __len__(self):
        return len(self.folderList)


    def __getitem__(self, index):
        tube = {'rgb' : torch.zeros((1, 3, 64, 224, 224)), 'of': torch.zeros((1, 2, 64, 224, 224)), 'joints' : np.zeros((64,17,3))}

        # Step 1 - iterate over training list
        with open(self.folderList, 'r') as file:
            sample = list(csv.reader(file))[index]
        
        try:
            # Step 2 - check if preprocessed data for sample exists
            path, joint_info, h, w = self.prepare(sample, self.frameDir, self.jsonDir)
            tube["scene"] = torch.zeros((1, 3, 64, h, w))    


            # Step 3 - Extract Tube & Augment if necessary
            res = self.extractTube(path, sample, h, w, joint_info)
            if res == "ERR":
                return res

            tube["rgb"], tube["scene"], tube["joints"], bb = res[0], res[1], res[2], res[3]


            # Step 4 - Extract Flow Tube
            tube["of"] = self.extractFlow(path, tube["scene"], h,  w, bb)

            np.save(path + "/joint/ehpi.npy", tube["joints"])

            # Step 5 - Extract EHPI
            # func call
        except Exception as e:
            print("Exception!!!!", e)
            return "EXCEPT"

        # step 6 - return tube
        return tube