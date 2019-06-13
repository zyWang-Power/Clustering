import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

#train_path = '/home/t1070/Desktop/FeatureLearningRotNet-master/datasets/MINIST_FOLDER/train/'
#for i in range(4):
#    if not os.path.exists('./dataset/train/'+str(i)):
#        os.mkdir('./dataset/train/'+str(i))
#    if not os.path.exists('./dataset/test/'+str(i)):
#        os.mkdir('./dataset/test/'+str(i))
#
#train_dir  = os.listdir(train_path)
#for c in train_dir:
#    train_path_1 = train_path+c
#    class_dir = os.listdir(train_path_1)
#    for pic in class_dir:
#        pic_path = train_path_1+'/'+pic
#        I =cv2.imread(pic_path)
#        I90 = np.rot90(I)
#        I180 = np.rot90(I90)
#        I270 = np.rot90(I180)
#        cv2.imwrite('./dataset/train/0/'+pic,I)
#        cv2.imwrite('./dataset/train/1/'+pic,I90)
#        cv2.imwrite('./dataset/train/2/'+pic,I180)
#        cv2.imwrite('./dataset/train/3/'+pic,I270)
#
train_path = '/home/t1070/Desktop/FeatureLearningRotNet-master/datasets/MINIST_FOLDER/test/'
for i in range(4):
    if not os.path.exists('./dataset/train/'+str(i)):
        os.mkdir('./dataset/train/'+str(i))
    if not os.path.exists('./dataset/test/'+str(i)):
        os.mkdir('./dataset/test/'+str(i))

train_dir  = os.listdir(train_path)
for c in train_dir:
    train_path_1 = train_path+c
    class_dir = os.listdir(train_path_1)
    for pic in class_dir:
        pic_path = train_path_1+'/'+pic
        I =cv2.imread(pic_path)
        I90 = np.rot90(I)
        I180 = np.rot90(I90)
        I270 = np.rot90(I180)
        cv2.imwrite('./dataset/test/0/'+pic,I)
        cv2.imwrite('./dataset/test/1/'+pic,I90)
        cv2.imwrite('./dataset/test/2/'+pic,I180)
        cv2.imwrite('./dataset/test/3/'+pic,I270)




