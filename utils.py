from datetime import datetime
import torch,pickle,math
from tqdm import tqdm
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.utils import save_image
from mmd import mmd_rbf
import torch
from PIL import Image
import torchvision.transforms as transform
from data_transform import  BatchSize,DeNormalize

PKL = r'./10_60.pkl'
classnum =10
outputdim = 60

# loader使用torchvision中自带的transforms函数
loader = transform.Compose([
    transform.ToTensor()])

normalize = transform.Compose([
    transform.Normalize(mean=[0.5], std=[0.5])
])

Transformations1 =transform.Compose([
        transform.RandomRotation((-10,10), resample=False, expand=False, center=None)
])

Transformations2 = transform.RandomChoice([
            transform.RandomRotation((-5,5), resample=False, expand=False, center=None),
            transform.RandomHorizontalFlip(p=0.5),
            transform.RandomVerticalFlip(p=0.5),
            transform.RandomAffine(degrees=(0, 0), translate=(0,0.1), scale=(0.9,1.1), shear=None, resample=False,
                           fillcolor=0),
            # transform.RandomGrayscale(p=0.5),
            # transform.ColorJitter(0.05, 0.05, 0.05, 0.05), #HSV以及对比度变化
])

Transformations3 =transform.Compose([
        transform.RandomAffine(degrees=(0,0), translate=(0,0.1), scale=None, shear=(-10,5), resample=False,
                            fillcolor=0)
])

Transformations4 =transform.Compose([
        transform.RandomAffine(degrees=(-10,10), translate=None, scale=(0.9,1.1), shear=(-10,5), resample=False,
                            fillcolor=0)
])

unloader = transform.ToPILImage()

os.environ['CUDA_VISIBLE_DEVICES']='0'
device_ids = [0]

def generator_noise(batchsize,out_dim):
    mean = np.zeros(out_dim)
    cov = np.eye(out_dim)
    noise = np.random.multivariate_normal(mean,cov,batchsize)
    return noise

def read_pkl(PKL):
    f = open(PKL,'rb')
    a = pickle.load(f)
    f.close()
    return a

def map_center(output,ui):
    total = output.shape[0]
    map_pedcc = torch.Tensor([]).cuda()
    for i in range(total):
        min_dis = 1000000000000
        for ui_50D_label in ui.values():
            distance = sum(sum((ui_50D_label.float().cuda()-output[i])**2))
            if distance<min_dis:
                min_label = ui_50D_label.float().cuda()
                min_dis = distance
        map_pedcc = torch.cat((map_pedcc,min_label),0)
    return map_pedcc

def sobel_1vs1_1D(im,de,criterion):
    weight_x = np.array([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]])

    weight_y = np.array([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]])

    weight_x = torch.from_numpy(weight_x).float().cuda()
    weight_y = torch.from_numpy(weight_y).float().cuda()


    sobel_x = F.conv2d(im,weight=weight_x,stride=1,padding=1)
    sobel_y = F.conv2d(im,weight=weight_y,stride=1,padding=1)
    sobel_de_x = F.conv2d(de,weight=weight_x,stride=1,padding=1)
    sobel_de_y = F.conv2d(de,weight=weight_y,stride=1,padding=1)

    loss_x = criterion(sobel_x,sobel_de_x)
    loss_y = criterion(sobel_y,sobel_de_y)

    return loss_x+loss_y

def train_nolabel(net1, net2, train_data,epoch, optimizer_en, optimizer_de, criterion):
    if torch.cuda.is_available():
        net1 = torch.nn.DataParallel(net1, device_ids=device_ids)
        net2 = torch.nn.DataParallel(net2, device_ids=device_ids)
        net1 = net1.cuda()
        net2 = net2.cuda()
    prev_time = datetime.now()
    train_loss = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    train_loss4 = 0
    train_loss5 = 0

    net1 = net1.train()
    net2 = net2.train()

    map_dict = read_pkl(PKL)
    tensor_empty1 = torch.Tensor([]).cuda()
    for label_index in range(classnum):
        tensor_empty1 = torch.cat((tensor_empty1, map_dict[label_index].float().cuda()), 0)

    for im, label in tqdm(train_data, desc="Processing train data: "):
        im11 = torch.Tensor([])
        im12 = torch.Tensor([])
        im13 = torch.Tensor([])
        im14 = torch.Tensor([])
        for i in range(im.size(0)):
            im11 = torch.cat((im11, normalize(loader(Transformations1((unloader(im[i]))))).unsqueeze(0)), 0)
            im12 = torch.cat((im12, normalize(loader(Transformations2((unloader(im[i]))))).unsqueeze(0)), 0)
            im13 = torch.cat((im13, normalize(loader(Transformations3((unloader(im[i]))))).unsqueeze(0)), 0)
            im14 = torch.cat((im14, normalize(loader(Transformations4((unloader(im[i]))))).unsqueeze(0)), 0)
        im=im.cuda()
        im11 = im11.cuda()
        im12 = im12.cuda()
        im13 = im13.cuda()
        im14 = im14.cuda()
        output_classifier11 = net1(im11)
        output_classifier12 = net1(im12)
        output_classifier13 = net1(im13)
        output_classifier14 = net1(im14)
        output_classifier = net1(im)

        output_deconvt = net2(output_classifier)

        loss4 = 5*(criterion(output_classifier13, output_classifier14)  \
                +criterion(output_classifier12, output_classifier13)  \
                +criterion(output_classifier12, output_classifier11) \
                +criterion(output_classifier14, output_classifier))

        sigma = generator_noise(output_classifier.size(0), output_classifier.size(1))
        new_out = output_classifier + torch.from_numpy(sigma * 0.05 * (output_classifier.size(1) ** 0.5)).float().cuda()
        output_deconv = net2(new_out)

        loss2 = criterion(output_deconvt, im)

        loss3 = 0.05 * sobel_1vs1_1D(im, output_deconv, criterion)

        z_fake= torch.Tensor([]).cuda()
        z_real=torch.cat((output_classifier,output_classifier14),0)
        z_real = torch.cat((z_real, output_classifier11), 0)
        z_real = torch.cat((z_real, output_classifier13), 0)
        z_real = torch.cat((z_real, output_classifier12), 0)
        batchsize = output_classifier.size(0)

        for b in range(5*(batchsize // classnum)):
            z_fake = torch.cat((z_fake, tensor_empty1), 0)

        z_fake = torch.cat((z_fake, tensor_empty1[0:batchsize % classnum]))
        z_fake = torch.cat((z_fake, tensor_empty1[0:batchsize % classnum]))
        z_fake = torch.cat((z_fake, tensor_empty1[0:batchsize % classnum]))
        z_fake = torch.cat((z_fake, tensor_empty1[0:batchsize % classnum]))
        z_fake = torch.cat((z_fake, tensor_empty1[0:batchsize % classnum]))

        loss1 = 5*mmd_rbf( z_real, z_fake)

        loss = loss1 + loss2 + loss3 + loss4

        # backward
        optimizer_en.zero_grad()
        optimizer_de.zero_grad()
        loss.backward()
        optimizer_en.step()
        optimizer_de.step()
        train_loss += loss.item()
        train_loss1 += loss1.item()
        train_loss2 += loss2.item()
        train_loss3 += loss3.item()
        train_loss4 += loss4.item()

    curr_time = datetime.now()
    h, remainder = divmod((curr_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = " Time %02d:%02d:%02d" % (h, m, s)

    epoch_str = ("Epoch %d. "
                 % (epoch))
    Loss = ("Train Loss1: %f, Train Loss2: %f,Train Loss3: %f,Train Loss4: %f,Train Loss5: %f,"
            % (train_loss1 / len(train_data), train_loss2 / len(train_data), train_loss3 / len(train_data),
               train_loss4 / len(train_data), train_loss5 / len(train_data)))

    prev_time = curr_time
    if not os.path.isdir('./model/MNIST/60/DATA/'):
        os.makedirs('./model/MNIST/60/DATA/')
    if epoch % 10 == 0 and epoch != 0:
        torch.save(net1, './model/MNIST/60/DATA/encoder_' + str(epoch) + '.pth')
        torch.save(net2, './model/MNIST/60/DATA/decoder_' + str(epoch) + '.pth')
    f = open('./model/MNIST/60/DATA/en_de.txt', 'a+')
    print(" ")
    print(epoch_str + time_str)
    print(Loss + "---------------")
    f.write(epoch_str + time_str + '\n')
    f.write(Loss + '\n')
    f.close()

