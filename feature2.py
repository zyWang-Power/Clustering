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
from utils import classnum,read_pkl,PKL
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES']='0'
device_ids = [0]

inver_transform2 = transform.Compose([
     DeNormalize([0.5], [0.5])
])



def train_feature(net1, train_data):
    #
    if torch.cuda.is_available():
        net1 = torch.nn.DataParallel(net1, device_ids=[0])
        net1 = net1.cuda()
    #
    for i_dir in range(classnum):
        if not os.path.isdir('./data/'+str(i_dir)):
            os.makedirs('./data/'+str(i_dir))
    label_np = np.array([0,0,0,0,0,0,0,0,0,0]*10).reshape(10,10)

    #
    label2=[]
    idx2=[]
    for im, label in tqdm(train_data,desc="Processing train data: "):
        # print(label)
        im = im.cuda()
        _,feat= net1(im)
        for i in range(feat.size(0)):
            distance = feat[i].cpu().numpy().tolist()
            idx = distance.index(max(distance))
            save_image(inver_transform2(im[i]),'./data/'+str(idx)+'/'+str(random.randint(1,10000000))+'.png')
            # MATRIX
            label_np[idx][label[i].item()] +=1
            #
            label2.append(idx)
        label1=label.numpy()
        for i in label1:
            idx2.append(i)

    t2 = np.array(idx2)
    t1 = np.array(label2)
    print('acc=%.4f, nmi=%.4f, ari=%.4f' % (
        metrics.acc(t1, t2), metrics.nmi(t1, t2), metrics.ari(t1, t2)))
    ##############################
    np.save(File +str(feat.size(1))+'_'+'.npy',label_np)

num = 70
File = './model/MNIST/60/MSE/'
net1 = torch.load(File + 'encoder_'+str(num)+'.pth')###############
for i in net1.parameters():
    i.requires_grad=False

train_feature(net1, L_test_data_1)

# acc=0.9862, nmi=0.9625, ari=0.9696