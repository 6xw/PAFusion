
import math
import os
import torch
from torch.nn.common_types import T
import torch.nn as nn
from collections import namedtuple
from torchvision import models
import torch.nn.functional as F
from einops import pack, rearrange, repeat, unpack
import thop
import time


class Convlayer(nn.Module):
    def __init__(self,inchannel,outchannel,kernelsize=3,stride=1,pooling=1,padding_mode='zeros', islrelu=False) -> None:
        super(Convlayer, self).__init__()
        self.incha = inchannel
        self.outcha = outchannel

        self.conv = nn.Conv2d(self.incha,self.outcha,kernelsize,stride,pooling,padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(self.outcha)
        if  islrelu:
            self.relu = nn.LeakyReLU(0.1)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out 

class Generator_(nn.Module):
    def __init__(self):  # in_ch,out_ch,ks,s,pad
        super(Generator_,self).__init__()
        #  encoder
        self.conv1 = Convlayer(6,64,padding_mode='zeros')
        self.conv2 = Convlayer(64,64,padding_mode='zeros')
        self.conv3 = Convlayer(128,64,padding_mode='zeros')
        # self.conv4 = Convlayer(192,64,padding_mode='zeros')
        #  decoder
        # self.conv6 = Convlayer(512,256)
        # self.conv7 = Convlayer(256,128,padding_mode='zeros')
        self.conv8 = Convlayer(192,64,padding_mode='zeros')
        self.conv9 = nn.Conv2d(64,3,3,1,1,padding_mode='zeros')

        # for m in self.modules():
        # # for mm in [self.conv1,self.conv2,self.conv3,self.conv4,self.conv5]:
        #     #  for m in mm:
        #         if isinstance(m, (nn.Conv2d)):
        #             print('init...')
        #             nn.init.kaiming_normal_(m.weight.unsqueeze(0), mode='fan_out', nonlinearity='relu')
        #             nn.init.kaiming_normal_(m.bias.unsqueeze(0), mode='fan_out', nonlinearity='relu')
        #             # nn.init.orthogonal_(m.weight.data.unsqueeze(0),gain=1)
        #             # nn.init.orthogonal_(m.bias.data.unsqueeze(0),gain=1)
        #             # nn.init.xavier_normal_(m.weight.data.unsqueeze(0))
        #             # nn.init.xavier_normal_(m.bias.data.unsqueeze(0))
        #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #             nn.init.constant_(m.weight, 1)
        #             nn.init.constant_(m.bias, 0)

    def forward(self,a,b):
        x = torch.cat((a,b),1)
        A1 = self.conv1(x)
        A2 = self.conv2(A1)
        A3 = self.conv3(torch.cat((A1,A2),1))
        # A4 = self.conv4(torch.cat((A1,A2,A3),1))
        x = torch.cat((A1,A2,A3),1)
        # x = self.conv6(x)
        # x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        return x
    
if __name__ == '__main__':
    model = Generator_()  # 
    # input_size = (1, 3, 480, 640)  # 