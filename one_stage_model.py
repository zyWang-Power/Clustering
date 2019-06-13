from Basic_blocks import *

class resnet14(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(resnet14,self).__init__()

        self.block1 = nn.Conv2d(in_channels,16,3,1)

        self.block2 = nn.Sequential(
            BasicBlock(16,16),
            BasicBlock(16,16)
        )

        self.block3 = nn.Sequential(
            BasicBlock(16,32,False),
            BasicBlock(32,32)
        )

        self.block4 = nn.Sequential(
            BasicBlock(32,64,False),
            BasicBlock(64,64),
            nn.AvgPool2d(8)
        )

        self.classifier = nn.Linear(64,num_classes)

        def forward(self, x):
            x = self.block1(x)

            x = self.block2(x)

            x = self.block3(x)

            x = self.block4(x)

            x = x.view(x.shape[0], -1)
            x = self.classifier(x)
            # x = Normal_scale(x)
            return x

class resnet20(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(resnet20, self).__init__()

        self.block1 = nn.Conv2d(in_channels, 16, 3, 1)

        self.block2 = nn.Sequential(
            BasicBlock(16, 16),
            BasicBlock(16, 16),
            BasicBlock(16, 16)
        )

        self.block3 = nn.Sequential(
            BasicBlock(16, 32, False),
            BasicBlock(32, 32),
            BasicBlock(32, 32)
        )

        self.block4 = nn.Sequential(
            BasicBlock(32, 64, False),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            nn.AvgPool2d(8)
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x=self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x=x.view(x.shape[0],-1)
        x=self.classifier(x)
        # x = Normal_scale(x)

        return x

class vae(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(vae,self).__init__()

        self.block1 = nn.Conv2d(in_channels, 32, 3, 1)

        self.block2 = nn.Sequential(
            BasicBlock(32, 64,False),
            BasicBlock(64, 128,False),
            BasicBlock(128, 256,False),
        )
        self.ave_pooling=nn.AvgPool2d(4)

        self.Linear_down=nn.Linear(256*4*4,out_channels)
        self.Linear_up=nn.Linear(out_channels,256*4*4)

        self.deconvBlock1 = nn.Sequential(
            DecodeBlock(256, 128)
        )
        self.deconvBlock2 = nn.Sequential(
            DecodeBlock(128, 64)
        )
        self.deconvBlock3 = nn.Sequential(
            DecodeBlock(64, 32)
        )
        self.conv1 = nn.Conv2d(32, 3, 1)


    def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)
        out1 = out.view(out.shape[0], -1)
        out_cla = self.Linear_down(out1)

        out2 = self.Linear_up(out_cla)
        out_up = out1.view(out.shape)

        out2 = self.deconvBlock1(out_up)
        out2 = self.deconvBlock2(out2)
        out2 = self.deconvBlock3(out2)
        out_con = self.conv1(out2)

        return out_cla, out_con

class SAE(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(SAE, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        self.block1 = nn.Sequential(
            nn.Linear(in_channels, 400),
            nn.ReLU(True)
        )

        self.Linear_down = nn.Linear(400, out_dim)

        self.deconvBlock1 = nn.Sequential(
            nn.Linear(in_channels, 400),
            nn.ReLU()
        )

        self.deconvBlock2 = nn.Sequential(
            nn.Linear(400, 28 * 28),
            nn.Sigmoid()
        )


    def encoder(self,x):
        x = x.view(-1, 28 * 28)
        x = self.block1(x)
        out = self.Linear_down(x)
        return out


    def decoder(self,out_dim):
        x=self.deconvBlock1(out_dim)
        x=self.deconvBlock2(x)
        return x


    def forward(self, x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x

