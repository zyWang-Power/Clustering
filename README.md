# Clustering <br>
**ICAE (An Image Clustering Auto-Encoder Based on Predefined Evenly-Distributed Class Centroids and MMD Distance)**

This is a reproducing code for ICAE [1]. ICAE is a method for clustering, specifically, ICAE is a image clustering auto-encoder based on predefined evenly-distributed class centroids and MMD distance. It can be applied to clustering to achieve the state-of-the-art results. The work is Zhngyong Wang completed during the period of study for a master's degree, in Shanghai University, China.

***

# Requirements <br>
You must have the following already installed on your system. <br><br>

1、Pytorch 1.0 <br>
2、sklearn <br>
3、python 3.6 <br>

***

# Quick start <br>
For reproducing the experiments on MNIST、Fashion-Mnist、COIL20 datasets in [1], run the following codes. <br><br>

1、python PEDCC.py : to Initialize the PEDCC, You need to set the cluster number, and every kind of dimension. We suggest that the MNIST every picture extract 60 dimension feature vector. <br>
2、Modify data_transform.py: you should choose datasets. <br>
3、python main.py for training. <br>
4、python feature2.py to calculate ACC and NMI. <br>
5、python generate_picture.py to generate each class clustering of images by pre-defined PEDCC centers. <br>

***

# Paper <br>

[1] Qiuyu Zhu, Zhengyong Wang. An Image Clustering Auto-Encoder Based on Predefined Evenly-Distributed Class Centroids and MMD Distance. Available at https://arxiv.org/abs/1906.03905

<br>

***

## If you have any questions, you can email me by zywang@shu.edu.cn.