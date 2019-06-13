from Basic_blocks import *
import torch

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

    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x.view(x.shape[0], -1)
        x=self.Linear_down(x)
        return x

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

    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x.view(x.shape[0], -1)
        x=self.Linear_down(x)
        return x

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
           # BasicBlock(64, 64),
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
        # x = torch.tanh(x)
        return x


