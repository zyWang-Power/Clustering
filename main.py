from one_stage_model import *
from two_stage_model import *
from data_transform import *
from utils import train_nolabel

base_lr = 0.001
epoches = 401
# lr_step = 70
outputdim = 60
dim = 1

net1 = encoder_plus_add(dim,outputdim)
net2 = decoder_plus_add(outputdim)

optimizer1 = torch.optim.Adam(net1.parameters(), lr=base_lr)
optimizer2 = torch.optim.Adam(net2.parameters(), lr=base_lr)

criterion = nn.MSELoss()

# def adjust_lr(optimizer, epoch):
#     lr = base_lr*(0.1**(epoch//lr_step))
#     for parameter in optimizer.param_groups:
#         parameter['lr'] = lr

print(" ####Start training  ####")

for epoch in range(epoches):
    train_nolabel(net1,net2,L_train_data_1,L_test_data_1,epoch,optimizer1,optimizer2,criterion)

print("Done!")
