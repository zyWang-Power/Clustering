import random,os,torch
import torch.utils.data as dataf
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10,MNIST
from torchvision.datasets import ImageFolder
import torchvision.transforms as transform
import numpy as np
import torch.nn.functional as F
from PIL import Image
from data_read import ImageFolder_L

BatchSize = 100

class DeNormalize(object):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t,m,s in zip(tensor,self.mean,self.std):
            t.mul_(s).add_(m)
        return tensor

class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return F.normalize(tensor, self.mean, self.std)

################################# MNIST #################################
inver_transform_MNIST = transform.Compose([
    DeNormalize([0.5],[0.5]),
    lambda x: x.cpu().numpy()*255.,
])

data_transform = transform.Compose([
    transform.Pad(padding=2,fill=0),
    transform.ToTensor(),
    transform.Normalize(mean=[0.5],std=[0.5])
])
MNIST_TRAIN = r"./dataset_equality/train"
MNIST_TEST = r"./dataset_equality/test"
# MNIST_TRAIN = r"./fashionMNIST_img/train"
# MNIST_TEST = r"./fashionMNIST_img/test"
#
L_train_set_1 = ImageFolder_L(MNIST_TRAIN,transform=data_transform)
L_test_set_1 = ImageFolder_L(MNIST_TEST,transform=data_transform)
L_train_data_1 = DataLoader(L_train_set_1,batch_size=BatchSize,shuffle=True)
L_test_data_1 = DataLoader(L_test_set_1,batch_size=BatchSize,shuffle=True)


# # ################################# COIL20 #################################
# # # #
# #
# inver_transform_COL20 = transform.Compose([
#     DeNormalize([0.5],[0.5]),
#     lambda x: x.cpu().numpy()*255.,
# ])
#
# data_transform_COL20 = transform.Compose([
#     transform.ToTensor(),
#     transform.Normalize(mean=[0.5],std=[0.5])
# ])
#
# COL20_TRAIN = r"./COIL20/train"
# COL20_TEST = r"./COIL20/test"
#
# COL20_train_set_1 = ImageFolder_L(COL20_TRAIN,transform=data_transform_COL20)
# COL20_test_set_1 = ImageFolder_L(COL20_TEST,transform=data_transform_COL20)
# COL20_train_data_1 = DataLoader(COL20_train_set_1,batch_size=BatchSize,shuffle=True)
# COL20_test_data_1 = DataLoader(COL20_test_set_1,batch_size=BatchSize,shuffle=True)
