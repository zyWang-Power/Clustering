# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
from data_transform import *
import pickle
import torch.nn as nn
import two_stage_model
import cv2
from datetime import datetime
from tqdm import tqdm
from torchvision.utils import save_image
import random
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]

def generator_noise(batchsize,out_dim):
    mean = np.zeros(out_dim)
    cov = np.eye(out_dim)
    noise = np.random.multivariate_normal(mean,cov,batchsize)
    return noise

def generate_data_net(net2,data_num,num):  # 3D
    map_dict ='./10_60.pkl'  # 读取PEDCC
    f = open(map_dict, 'rb')
    mean_var_40 = pickle.load(f)  # 取出
    f.close()
    data_40=[]

    cnt = 0
    for keys, values in mean_var_40.items():#这里面keys表示编号
        if cnt == 0:
            data_40 = values
            cnt += 1
        else:
            data_40 = np.concatenate([data_40, values], 0)  # 一层一层叠加
            cnt += 1

    with torch.no_grad():
        cnt2 = 0

        for j in range(10):
                input1=torch.from_numpy(np.array([data_40[cnt2]])).float()
                out2 = net2(input1)
                ii =inver_transform_MNIST(out2[0])[0]
                ii[ii<0]=0
                ii[ii> 255] = 255
                cnt2+=1
                plt.subplot(data_num,data_num,cnt2)
                plt.imshow(ii.astype(np.uint8),cmap=plt.cm.gray)
                plt.axis('off')
                plt.savefig('./model/MNIST/60/MSE/'+str(num)+'.png')
        plt.show()

num= 70
data_num =10
net2 = torch.load(r'./model/MNIST/60/MSE/decoder_'+str(num)+'.pth')

for i in net2.parameters():
    i.requires_grad=False

net2 = net2.cuda()
net2 = net2.eval()
generate_data_net(net2,data_num,num)

