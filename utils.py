from datetime import datetime
import torch,pickle,math
import torch.nn as nn
from torch.nn import Parameter
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
import metrics
import torchvision.transforms as transform
from data_transform import BatchSize,DeNormalize
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES']='0'
device_ids = [0]

classnum = 10
outputdim = 60
PKL = './' + str(classnum) + '_' + str(outputdim) + '.pkl'

class Softmax_PEDCC(nn.Module):
    def __init__(self, in_features, out_features, PKL):
        super(Softmax_PEDCC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #
        self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=False)
        #
        map_dict = read_pkl(PKL)
        tensor_empty = torch.Tensor([]).cuda()
        #
        for label_index in range(self.out_features):
            tensor_empty = torch.cat((tensor_empty, map_dict[label_index].float().cuda()), 0)
        label_40D_tensor = tensor_empty.view(-1, self.in_features).permute(1, 0)
        label_40D_tensor = label_40D_tensor.cuda()
        self.weight.data = label_40D_tensor


    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features
        cos_theta = x.mm(w)  # size=(B,Classnum)  x.dot(ww)
        return cos_theta  # size=(B,Classnum,1)


loader = transform.Compose([
    transform.ToTensor()])

normalize = transform.Compose([
    transform.Normalize(mean=[0.5], std=[0.5])
    ])

Transformations1 =transform.Compose([
        transform.RandomRotation((-10,10), resample=False, expand=False, center=None)
])

Transformations2 = transform.RandomChoice([
        transform.RandomHorizontalFlip(p=1)
])

Transformations3 =transform.Compose([
        transform.RandomAffine(degrees=(0,0), translate=(0,0.1), scale=None, shear=None, resample=False,
                            fillcolor=0)
])

Transformations4 =transform.Compose([
        transform.RandomAffine(degrees=(0,0), translate=None, scale=(0.9,1.1), shear=None, resample=False,
                            fillcolor=0)
])

Transformations5 =transform.Compose([
        transform.RandomAffine(degrees=(0, 0), translate=None, scale=None, shear=(-10,5), resample=False,
                               fillcolor=0),
])

unloader = transform.ToPILImage()

grad_list = []  # 用于存放训练中中间梯度
grad_res = []  # 存放处理后的梯度
grad_png = []
grad_loss1 = []
grad_loss4 = []

def print_grad(grad):
    grad_list.extend(grad.cpu().numpy())


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

def sobel_1vs1_1D(im,de,criterion):
    weight_x = np.array([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]])

    weight_y = np.array([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]])

    weight_x = torch.from_numpy(weight_x).float().cuda()
    weight_y = torch.from_numpy(weight_y).float().cuda()


    sobel_x = F.conv2d(im,weight=weight_x,stride=1,padding=1)
    sobel_y = F.conv2d(im,weight=weight_y,stride=1,padding=1)
    sobel_de_x = F.conv2d(de,weight=weight_x,stride=1,padding=1)
    sobel_de_y = F.conv2d(de,weight=weight_y,stride=1,padding=1)

    loss_x = criterion(sobel_de_x,sobel_x)
    loss_y = criterion(sobel_de_y,sobel_y)

    return loss_x+loss_y

def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(
        q_logit, dim=-1)), 1)
    return torch.mean(_kl)

def test(net1, test_data):
    #
    if torch.cuda.is_available():
        net1 = torch.nn.DataParallel(net1, device_ids=[0])
        net1 = net1.cuda()
    #
    label2=[]
    idx2=[]
    for im, label in tqdm(test_data,desc="Processing train data: "):
        im = im.cuda()
        _,feat= net1(im)
        for i in range(feat.size(0)):
            distance = feat[i].cpu().numpy().tolist()
            idx = distance.index(max(distance))
            label2.append(idx)
        label1=label.numpy()
        for i in label1:
            idx2.append(i)
    t2 = np.array(idx2)
    t1 = np.array(label2)

    return metrics.acc(t2, t1), metrics.nmi(t2, t1)


def train_nolabel(net1, net2, train_data,test_data,epoch, optimizer_en, optimizer_de, criterion):
    if torch.cuda.is_available():
        net1 = torch.nn.DataParallel(net1, device_ids=device_ids)
        net2 = torch.nn.DataParallel(net2, device_ids=device_ids)
        net1 = net1.cuda()
        net2 = net2.cuda()
    #
    prev_time = datetime.now()
    #
    train_loss = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    train_loss4 = 0
    train_loss5 = 0
    #
    net1 = net1.train()
    net2 = net2.train()
    #
    map_dict = read_pkl(PKL)
    tensor_empty_MMD = torch.Tensor([]).cuda()
    for label_index in range(classnum):
        tensor_empty_MMD = torch.cat((tensor_empty_MMD, map_dict[label_index].float().cuda()), 0)
    #
    for im, label in tqdm(train_data, desc="Processing train data: "):
        im11 = torch.Tensor([])
        im12 = torch.Tensor([])
        im13 = torch.Tensor([])
        im14 = torch.Tensor([])
        im15 = torch.Tensor([])

        for i in range(im.size(0)):
            im11 = torch.cat((im11, normalize(loader(Transformations1((unloader(im[i]))))).unsqueeze(0)), 0)
            im12 = torch.cat((im12, normalize(loader(Transformations2((unloader(im[i]))))).unsqueeze(0)), 0)
            im13 = torch.cat((im13, normalize(loader(Transformations3((unloader(im[i]))))).unsqueeze(0)), 0)
            im14 = torch.cat((im14, normalize(loader(Transformations4((unloader(im[i]))))).unsqueeze(0)), 0)
            im15 = torch.cat((im15, normalize(loader(Transformations5((unloader(im[i]))))).unsqueeze(0)), 0)


        im=im.cuda()
        im11 = im11.cuda()
        im12 = im12.cuda()
        im13 = im13.cuda()
        im14 = im14.cuda()
        im15 = im15.cuda()

        # net output
        output_classifier11,output_classifier11_softmax = net1(im11)
        output_classifier12,output_classifier12_softmax = net1(im12)
        output_classifier13,output_classifier13_softmax = net1(im13)
        output_classifier14,output_classifier14_softmax = net1(im14)
        output_classifier15,output_classifier15_softmax = net1(im15)

        output_classifier,output_classifier_softmax = net1(im)
        output_no_noise = net2(output_classifier)
        # KLD
        loss1 = 1*(kl_categorical(output_classifier13_softmax, output_classifier_softmax)
         + kl_categorical(output_classifier12_softmax, output_classifier_softmax)
         + kl_categorical(output_classifier11_softmax, output_classifier_softmax)
         + kl_categorical(output_classifier14_softmax, output_classifier_softmax)
         + kl_categorical(output_classifier15_softmax, output_classifier_softmax))


        # mse
        loss5 = 0.01*(criterion(output_classifier13, output_classifier14)  \
                +criterion(output_classifier12, output_classifier13)  \
                +criterion(output_classifier12, output_classifier11) \
                +criterion(output_classifier14, output_classifier)  \
                +criterion(output_classifier15, output_classifier14))

        ###############################################################
        # add noise --mse  boundary
        sigma = generator_noise(output_classifier.size(0), output_classifier.size(1))
        new_out = output_classifier + torch.from_numpy((0.05 * sigma) / ((output_classifier.size(1) ** 0.5))).float().cuda()
        output_noise = net2(new_out)  #
        ###############################################################
        # decoder no noise --mse
        loss2 = criterion(output_no_noise , im)
        ###############################################################
        loss3 = 0.05 * sobel_1vs1_1D(im, output_noise , criterion)
        #
        z_Pedcc= torch.Tensor([]).cuda()
        #
        z_real = output_classifier

        # use hook to see gradient
        # z_real.register_hook(print_grad)
        #
        batchsize = output_classifier.shape[0]
        for b in range(1*(batchsize // classnum)):
            z_Pedcc = torch.cat((z_Pedcc, tensor_empty_MMD), 0)

        z_Pedcc = torch.cat((z_Pedcc, tensor_empty_MMD[0:batchsize % classnum]))

        # MMD weight
        ###############################################################
        loss4 = 1* mmd_rbf(z_real, z_Pedcc)
        z_real.register_hook(print_grad)
        # loss
        loss =  loss5 + loss3 + loss4 + loss2
        # backward
        optimizer_en.zero_grad()
        optimizer_de.zero_grad()
        #
        loss.backward()
        #
        optimizer_en.step()
        optimizer_de.step()
        #
        train_loss += loss.item()
        train_loss1 += loss1.item()
        train_loss2 += loss2.item()
        train_loss3 += loss3.item()
        train_loss4 += loss4.item()
        train_loss5 += loss5.item()
    #     # deal grad
    #     x_norm = np.linalg.norm(np.array(grad_list),ord=None,axis=1,keepdims=False).sum()
    #     # print(x_norm)
    # #
    # x_norm=x_norm / np.array(grad_list).shape[0]*BatchSize
    # grad_res.append(x_norm)
    # grad_list.clear()
    #
    curr_time = datetime.now()
    h, remainder = divmod((curr_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = " Time %02d:%02d:%02d" % (h, m, s)

    epoch_str = ("Epoch %d. " % (epoch))
    Loss = ("Train Loss1: %f, Train Loss2: %f,Train Loss3: %f,Train Loss4: %f,"
            "Train Loss5: %f," % (train_loss1 / len(train_data),
                                  train_loss2 /len(train_data),
                                  train_loss3 / len(train_data),
                                  train_loss4 / len(train_data),
                                  train_loss5 / len(train_data)))
    grad_loss1.append(train_loss1 / len(train_data))
    grad_loss4.append(train_loss4 / len(train_data))
    ###############  test  #############
    torch.save(net1, './model_encoder.pth')
    net11 = torch.load('./model_encoder.pth')
    for i in net11.parameters():
        i.requires_grad = False
    Acc,NMI = test(net11, test_data)
    grad_png.append(Acc)
    ###############  test  #############
    # build dir
    File = './model/MNIST/60/MSE1/'
    if not os.path.isdir(File):
        os.makedirs(File)

    if epoch % 2 == 0 and epoch != 0:
        res_plot = np.array(grad_loss1)
        np.savetxt(File  + '_loss1_aug.txt', res_plot, fmt='%0.8f')
        plt.plot([i for i in range(len(grad_loss1))], grad_loss1)
        plt.savefig(File + '_loss1_aug.png')

        res_plot = np.array(grad_loss4)
        np.savetxt(File  + '_loss4_MMD.txt', res_plot, fmt='%0.8f')
        #
        plt.plot([i for i in range(len(grad_loss4))], grad_loss4)
        plt.savefig(File + '_loss4_MMD.png')

        res_plot = np.array(grad_png)
        np.savetxt(File  + '_loss_ACC.txt', res_plot, fmt='%0.8f')
        #
        plt.plot([i for i in range(len(grad_png))], grad_png)
        plt.savefig(File + '_loss_ACC.png')


    # save encoder and decoder
    if epoch % 10 == 0 and epoch != 0:
        torch.save(net1, File + 'encoder_' + str(epoch) + '.pth')
        torch.save(net2, File + 'decoder_' + str(epoch) + '.pth')
        # plt.show()batch100_lr0.001_aug5_MMD_0.001_5_version1.txt
    # write log
    f = open(File + 'log.txt', 'a')
    print(" ")
    print(epoch_str + time_str )
    print(Loss)
    print('Clustering ACC = %.4f,Clustering NMI = %.4f' % (Acc, NMI))

    print("---------------")
    f.write(epoch_str + time_str + '\n')
    f.write(Loss + '\n')
    f.write("Acc : " + str(Acc) + ',' + ' NMI :' + str(NMI)+'\n')
    f.close()
    #


