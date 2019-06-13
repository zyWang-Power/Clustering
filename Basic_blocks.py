import torch.nn as nn
import torch.nn.functional as F


def conv_3x3(in_channel,out_channel,stride=1):
    "3x3 convolution with padding"
    return  nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)


class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel,same_shape=True):
        super(BasicBlock,self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = conv_3x3(in_channel,out_channel,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = conv_3x3(out_channel,out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if  not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel,out_channel,1,stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out),inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if not self.same_shape:
            x=self.conv3(x)

        return F.relu(x + out,inplace=True)


class DecodeBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DecodeBlock,self).__init__()

        self.deconv1 = nn.ConvTranspose2d(in_channel,in_channel//2,2,stride= 2)
        self.conv=nn.Sequential(
                nn.Conv2d(in_channel//2,out_channel,3,padding=1,bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True))

    def forward(self,x):
        out = self.deconv1(x)
        out = self.conv(out)
        return out