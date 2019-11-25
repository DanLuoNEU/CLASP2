import torch
import torch.nn as nn
import torch.nn.functional as F

class EHPI_Classifier(nn.Module):
    def __init__(self, num_classes, gpu_id ):
        super(EHPI_Classifier,self).__init__()
        self.num_classes = num_classes;

        self.conv1 = nn.Conv2d(, , kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(20)

        self.conv2 = nn.Conv2d(, , kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(20)

        self.maxpool1 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(, , kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(20)

        self.conv4 = nn.Conv2d(, , kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(20)

        self.maxpool2 = nn.MaxPool2d(2,2)

        self.conv5 = nn.Conv2d(, , kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(20)

        self.conv6 = nn.Conv2d(, , kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(20)
        
        self.averagepool = nn.AvgPool2d()

        self.fc = nn.Linear(, num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x));
        x = self.conv1_bn(x);

        x = F.relu(self.conv2(x));
        x = self.conv2_bn(x);

        x = self.maxpool1(x)

        x = F.relu(self.conv3(x));
        x = self.conv3_bn(x);

        x = F.relu(self.conv4(x));
        x = self.conv4_bn(x);

        x = self.maxpool2(x)

        x = F.relu(self.conv5(x));
        x = self.conv5_bn(x);

        x = F.relu(self.conv6(x));
        x = self.conv6_bn(x);

        x = self.averagepool(x);

        x = self.fc(x)
        return x