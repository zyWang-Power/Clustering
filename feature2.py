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
import metrics
from utils import classnum

os.environ['CUDA_VISIBLE_DEVICES']='0'
device_ids = [0]

inver_transform2 = transform.Compose([
     DeNormalize([0.5], [0.5])
])

def read_pkl():
    f = open(r'./10_60.pkl','rb')####################
    a = pickle.load(f)
    f.close()
    return a


def train_feature(net1, train_data):
    map_dict = read_pkl()
    if torch.cuda.is_available():
        net1 = torch.nn.DataParallel(net1, device_ids=[0])
        net1 = net1.cuda()
    prev_time = datetime.now()

    for i_dir in range(classnum):
        if not os.path.isdir('./data/'+str(i_dir)):
            os.makedirs('./data/'+str(i_dir))
    label_np = np.array([0,0,0,0,0,0,0,0,0,0]*10).reshape(10,10)
    # label_np = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 20).reshape(20, 20)

    label2=[]
    idx2=[]
    for im, label in tqdm(train_data,desc="Processing train data: "):
        im = im.cuda()
        feat = net1(im)
        for i in range(feat.size(0)):
            distance_list = list()
            for ui_50D_label in map_dict.values():
                distance = sum(sum((ui_50D_label.float().cuda()-feat[i])**2))
                distance_list.append(distance.item())
            idx = distance_list.index(min(distance_list))
            save_image(inver_transform2(im[i]),'./data/'+str(idx)+'/'+str(random.randint(1,10000000))+'.png')
            label_np[idx][label[i].item()] +=1
            label2.append(idx)
        label1=label.numpy()
        # for _,i in enumerate(label):
        #     idx2.append(i)
        for i in label1:
            idx2.append(i)

    t2 = np.array(idx2)
    t1 = np.array(label2)
    # print(t2.shape)
    # t2 = t2.reshape([t1.size,-1]).squeeze(0)
    print('acc=%.4f, nmi=%.4f, ari=%.4f' % (
        metrics.acc(t1, t2), metrics.nmi(t1, t2), metrics.ari(t1, t2)))
    
    corr_num = 0
    for item in label_np:
        corr_num += item.max()
    corr = corr_num/label_np.sum()
    print (corr)
    np.save('./model/MNIST/feature/'+str(feat.size(1))+'_'+'.npy',label_np)

num =10
net1 = torch.load(r'./model/MNIST/60/DATA/encoder_'+str(num)+'.pth')###############
for i in net1.parameters():
    i.requires_grad=False

train_feature(net1, L_test_data_1)


