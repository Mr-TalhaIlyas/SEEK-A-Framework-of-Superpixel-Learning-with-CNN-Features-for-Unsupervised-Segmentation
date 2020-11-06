
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(3)  # choose GPU:0
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from tqdm import trange
import cv2
import numpy as np
np.random.seed(1943)
from skimage import segmentation
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.cuda.manual_seed_all(1943)



class SE(nn.Module):

    def __init__(self, n_features, reduction=8):
        super(SE, self).__init__()

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # get the filter of same size as them iamge hight or width
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4]) 
        y = y.permute(0, 2, 3, 1)
        y = self.relu(self.linear1(y))
        y = self.sigmoid(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        
        return y

class SEEK(nn.Module):

    def __init__(self, n_features):
        super(SEEK, self).__init__()

        # convolutions
        self.conv_in = nn.Conv2d(3, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_in = nn.BatchNorm2d(n_features)
        self.relu_in = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm1 = nn.BatchNorm2d(n_features)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(n_features, n_features//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(n_features//2)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(n_features//2, n_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm3 = nn.BatchNorm2d(n_features)
        self.relu3 = nn.ReLU(inplace=True)
        # squeeze and excitation

        self.sqex  = SE(n_features)

    def forward(self, x):
        
        x_in = self.relu_in(self.norm_in(self.conv_in(x)))
        # SE-Block 1
        y = self.relu1(self.norm1(self.conv1(x_in)))
        y = self.relu2(self.norm2(self.conv2(y)))
        y = self.relu3(self.norm3(self.conv3(y)))
        # squeezing and exciting
        y = self.sqex(y)
        # adding residuals
        y1 = torch.add(x_in, y)
        
        # # SE-Block 2
        y2 = self.relu1(self.norm1(self.conv1(y)))
        y2 = self.relu2(self.norm2(self.conv2(y2)))
        y2 = self.relu3(self.norm3(self.conv3(y2)))
        # squeezing and exciting
        y2 = self.sqex(y2)
        # adding residuals
        y2 = torch.add(y1, y2)

        return y2




