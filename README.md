#Clustering
ICAE code(An Image Clustering Auto-Encoder Based on Predefined Evenly-Distributed Class Centroids and MMD Distance)

This is a reproducing code for ICAE [1]. ICAE is a method for clustering, specifically, ICAE is a image clustering auto-encoder based on predefined evenly-distributed class centroids and MMD distance. It can be applied to clustering to achieve the state-of-the-art results. The work is Zhngyong Wang completed during the period of study for a master's degree,in Shanghai University,China.

***

#Requirements
You must have the following already installed on your system.
1、Pytorch 1.0
2、sklearn
3、python 3.6

***

#Quick start
For reproducing the experiments on MNIST、Fashion-Mnist、COIL20 datasets in [1], run the following codes.
1、python PEDCC.py : to Initialize the PEDCC, You need to set the cluster number, and every kind of dimension. We suggest that the MNIST every picture extract 60 dimension feature vector.
2、Modify data_transform.py: you should choose datasets.
3、python main.py for training.
4、python feature2.py to calculate ACC and NMI
5、python generate_picture.py to generate each class clustering of images by pre-defined PEDCC centers.

***

#Paper

[1] An Image Clustering Auto-Encoder Based on Predefined Evenly-Distributed Class Centroids and MMD Distance
Qiuyu Zhu, Zhengyong Wang. Available at https://arxiv.org/abs/1906.03905


##If you have any questions, you can email me by zywang@shu.edu.cn.