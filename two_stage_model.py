from Basic_blocks import *
import torch
from utils import PKL,Softmax_PEDCC,classnum,outputdim

class encoder_plus_add(nn.Module):
    def  __init__(self,in_channels,out_channels):
        super(encoder_plus_add,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.block2 = nn.Sequential(
            BasicBlock(32, 32, False),
        )

        self.block3 = nn.Sequential(
            BasicBlock(32, 64, False),
        )

        self.block4 = nn.Sequential(
            BasicBlock(64, 128, False),
        )
        self.block5 = nn.Sequential(
            BasicBlock(128, 256, False),
        )
        self.block6 = nn.Sequential(
            BasicBlock(256, 512, False),
        )
        self.Linear_down = nn.Linear(512*1*1, out_channels)

        self.out = Softmax_PEDCC(outputdim,classnum,PKL)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x.view(x.shape[0], -1)
        x = self.Linear_down(x)
        x1 = self.l2_norm(x)
        out = self.out(x1)
        return x1, out

class decoder_plus_add(nn.Module):
    def __init__(self,in_channels):
        super(decoder_plus_add,self).__init__()
        self.Linear_up = nn.Linear(in_channels, 512*1*1)

        self.deconvBlock6 = nn.Sequential(
            DecodeBlock(512, 256),
        )
        self.deconvBlock7 = nn.Sequential(
            DecodeBlock(256, 128),
        )
        self.deconvBlock1 = nn.Sequential(
            DecodeBlock(128, 64),
        )
        self.deconvBlock2 = nn.Sequential(
            DecodeBlock(64, 32),
        )
        self.deconvBlock3 = nn.Sequential(
            DecodeBlock(32, 32),
        )

        self.conv1 = nn.Conv2d(32, 1, 1)

    def forward(self,x):
        x = self.Linear_up(x)
        x = x.view(-1,512,1,1)
        x = self.deconvBlock6(x)
        x = self.deconvBlock7(x)
        x = self.deconvBlock1(x)
        x = self.deconvBlock2(x)
        x = self.deconvBlock3(x)
        x = self.conv1(x)
        return x

class encoder_plus_add128(nn.Module):
    def  __init__(self,in_channels,out_channels):
        super(encoder_plus_add128,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.block2 = nn.Sequential(
            BasicBlock(32, 32, False),
        )

        self.block3 = nn.Sequential(
            BasicBlock(32, 64, False),
        )

        self.block4 = nn.Sequential(
            BasicBlock(64, 128, False),
        )
        self.block5 = nn.Sequential(
            BasicBlock(128, 256, False),
        )
        self.block6 = nn.Sequential(
            BasicBlock(256, 512, False),
        )
        self.Linear_down = nn.Linear(512*4*4, out_channels)

        self.out = Softmax_PEDCC(outputdim, classnum, PKL)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x.view(x.shape[0], -1)
        x=self.Linear_down(x)
        x1 = self.l2_norm(x)
        out = self.out(x1)
        return x1, out

class decoder_plus_add128(nn.Module):
    def __init__(self,in_channels):
        super(decoder_plus_add128,self).__init__()
        self.Linear_up = nn.Linear(in_channels, 512*4*4)

        self.deconvBlock6 = nn.Sequential(
            DecodeBlock(512, 256),
        )
        self.deconvBlock7 = nn.Sequential(
            DecodeBlock(256, 128),
        )
        self.deconvBlock1 = nn.Sequential(
            DecodeBlock(128, 64),
        )
        self.deconvBlock2 = nn.Sequential(
            DecodeBlock(64, 32),
        )
        self.deconvBlock3 = nn.Sequential(
            DecodeBlock(32, 32),
        )
        self.conv1 = nn.Conv2d(32, 1, 1)

    def forward(self,x):
        x = self.Linear_up(x)
        x = x.view(-1,512,4,4)
        x = self.deconvBlock6(x)
        x = self.deconvBlock7(x)
        x = self.deconvBlock1(x)
        x = self.deconvBlock2(x)
        x = self.deconvBlock3(x)
        x = self.conv1(x)
        return x

class encoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(encoder,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True))

        self.block2 = nn.Sequential(
            BasicBlock(16, 32, False),
            BasicBlock(32, 64, False),
            BasicBlock(64, 64, False),
        )

        self.Linear_down = nn.Linear(64 * 4 * 4, out_channels)
        self.out = Softmax_PEDCC(outputdim, classnum, PKL)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x = x.view(x.shape[0], -1)
        x = self.Linear_down(x)
        x1 = self.l2_norm(x)
        out = self.out(x1)
        return x1, out


class decoder(nn.Module):
    def __init__(self,in_channels):
        super(decoder,self).__init__()
        self.Linear_up = nn.Linear(in_channels, 64*4*4)

        self.deconvBlock1 = nn.Sequential(
            DecodeBlock(64, 64)
        )
        self.deconvBlock2 = nn.Sequential(
            DecodeBlock(64, 32)
        )
        self.deconvBlock3 = nn.Sequential(
            DecodeBlock(32, 16)
        )
        self.conv1 = nn.Conv2d(16, 1, 1)

    def forward(self,x):
        x = self.Linear_up(x)
        x = x.view(-1,64,4,4)
        x = self.deconvBlock1(x)
        x = self.deconvBlock2(x)
        x = self.deconvBlock3(x)
        x = self.conv1(x)
        return x
