#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
#import Trainer
#from network import NFM
import torch.utils.data as Data
#from Utils.criteo_loader import getTestData, getTrainData

nfm_config = \
{
    'n_class':9,
    'linear_hidden1':2000,
    #'linear_hidden':100,#线性模型输出层（隐层个数）
    #'embed_input_dim':1001,#embed输入维度
    #'embed_dim': 100, # 用于控制稀疏特征经过Embedding层后的稠密特征大小，embed输出维度
    #'dnn_hidden_units': [100,11],#MLP隐层和输出层
    
    'dnn_hidden_units':[100,8],#MLP隐层
    'num_sparse_features_cols':10477,#the number of the gene columns
    'num_dense_features': 0,#dense features number
    'bi_dropout': 0.5,#Bi-Interaction 的dropout
    'num_epoch': 500,#训练epoch次数
    'batch_size': 16,#batch_size
    'lr': 1e-3,
    'l2_regularization': 1e-4,
    'device_id': 0,
    'use_cuda': False,
    'epoch':1000,
    
    #'train_file': '../Data/criteo/processed_data/train_set.csv',
    #'fea_file': '../Data/criteo/processed_data/fea_col.npy',
    #'validate_file': '../Data/criteo/processed_data/val_set.csv',
    #'test_file': '../Data/criteo/processed_data/test_set.csv',
    #'model_name': '../TrainedModels/NFM.model'
    #'train_file':'data/xiaoqiu_gene_5000/train/final_5000_encode_100x.csv',
    #'train_data':'dataset/qiuguan/encode/encode_1000/train/train_encode_data_1000_new.csv',
    #'train_label':'dataset/qiuguan/non_code/train/train_label.csv',
    #'guan_test_data':'dataset/qiuguan/non_code/guan_test/guan_test_data.csv',
    #'guan_test_label':'dataset/qiuguan/non_code/guan_test/guan_test_label.csv',
    #'test_data':'dataset/qiuguan/encode/encode_1000/test/test_encode_data_1000_new.csv',
    #'test_label':'dataset/qiuguan/non_code/test/test_labels.csv',
    #'title':'dataset/xiaoguan/RF/RF_for_train/train_class_9/test/test_data.csv',
    'gene_name':'dataset/qiuguan/origin_800/gene_name.csv',
    'label_name':'dataset/qiuguan/origin_800/gene_label.csv'
    #'all':''
    #'title':'data/xiaoqiu_gene_5000/train/gene_5000_gene_name.csv',
    #'all':'data/xiaoqiu_gene_5000/train/gene_5000_label_name.csv'
}
#model definition
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0=nn.BatchNorm1d(3300)
        self.fc1 = nn.Linear(3300, 2000)
        self.bn1= nn.BatchNorm1d(2000)
        self.fc2 = nn.Linear(2000, 100)
        self.bn2=nn.BatchNorm1d(100)
        self.fc3=nn.Linear(100,9)
        self.bn3=nn.BatchNorm1d(9)
        
        self.drop=nn.Dropout(0.5)
    def forward(self, x):
        x=self.bn0(x)
        x = F.relu(self.drop(self.bn1(self.fc1(x))))
        x = F.relu(self.drop(self.bn2(self.fc2(x))))
        return F.softmax(self.bn3(self.fc3(x)), dim=1)
model = MLP().cuda()
print(model)

import os
import time
import argparse
import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
#3from tensorboardX import SummaryWriter
import torch.nn.functional as F  # 激励函数的库
#import network
#import config
#import evaluate
#import data_utils
#import Trainer

def one_hot_smoothing(labels, classes, label_smoothing=0.2):
    #n = len(labels)
    n=labels.shape[0]
    eoff = label_smoothing / classes
    output = np.ones((n, classes), dtype=np.float32) * eoff
    for row, label in enumerate(labels):
        output[row, label] = 1 - label_smoothing + eoff
        #print("row:",row,"label:",label)
    return output

def one_hot(labels, classes):
    n = len(labels)
    #eoff = label_smoothing / classes
    output = np.zeros((n, classes), dtype=np.float32)
    for row, label in enumerate(labels):
        output[row, label] = 1
        #print("row:",row,"label:",label)
    return output


import pandas as pd 

from sklearn.model_selection import train_test_split




def one_hot_smoothing(labels, classes, label_smoothing=0.2):
    #n = len(labels)
    n=labels.shape[0]
    eoff = label_smoothing / classes
    output = np.ones((n, classes), dtype=np.float32) * eoff
    for row, label in enumerate(labels):
        output[row, label] = 1 - label_smoothing + eoff
        #print("row:",row,"label:",label)
    return output

def one_hot(labels, classes):
    n = len(labels)
    #eoff = label_smoothing / classes
    output = np.zeros((n, classes), dtype=np.float32)
    for row, label in enumerate(labels):
        output[row, label] = 1
        #print("row:",row,"label:",label)
    return output

class KZDatasetTest(data.Dataset):
    """ Construct the FM pytorch dataset. """
    #def __init__(self, file,label_file, feature_map,n_class=16):
    def __init__(self, csv_path):
    
        self.data_info = self.get_data_info(csv_path)
        
        
            
        
        
        

    def __getitem__(self, index):
        # Dataset读取图片的函数
        data, label = self.data_info[index]
        #img = Image.open(img_pth).convert('RGB')
        
        return data, label

    def __len__(self):
        return len(self.data_info)
    
    
    
    def get_data_info(self,csv_path):
        #解析路径
        #转为一维list存储，每一位为【图片路径，图片类别】
        labels=[]
        data_info=[]
        df=pd.read_csv(csv_path,sep=',',header=None)
        
        df=df.iloc[1:,1:]
        #print("df:",df)
        #print(df.iloc[:,-1])
        #df=df.applymap(ast.literal_eval)
        rows,cols=df.shape
        #print(rows,cols)
        for i in df.iloc[:,-1]:
            #print(i)
            labels.append(int(i))
        #print('labels:',labels)
        labels=np.array(labels)
        #print('labels:',labels)
        #labels=np.array(labels)
        labels=one_hot_smoothing(labels,nfm_config['n_class'])
        for i in range(rows):
            data=df.iloc[i,:-1]
            data=data.astype(float)##############
            #print("i,data:",i,data)
            #data=pd.DataFrame(data,dtype=float)###############
            data=np.array(data)##
            
            label=labels[i]
            #print(data.shape)
            #print(label.shape)
            #label=label.tolist()
            data=torch.from_numpy(data)#
            label=torch.from_numpy(label)#
           
            
            data_info.append((data,label))
        return data_info
    
    
    
    
import torch
import torch.nn as nn
from torch.utils.data.dataset import *
from PIL import Image
from torch.nn import functional as F
import random
from sklearn.model_selection import train_test_split
import ast
import torchvision



#######找特征基因#############从3301中找200个基因
#########################################################本次测试的目的是看200个基因的分类效果
##########测试步骤：从3301个基因中提取350个
############用200个构建新的分类模型
#################特征基因
######################为小球，根据上边的测试的基因个数，350最大
import torch

import torch.nn as nn
import torch.optim as optim


from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#from mnist_test import Net, train, test


# Network parameters
class Params(object):
    batch_size = 64
    test_batch_size = 20
    epochs = 5
    lr = 0.01
    momentum = 0.5
    no_cuda = True
    seed = 1
    log_interval = 10
    
    def __init__(self):
        pass

args = Params()
torch.manual_seed(args.seed)
#device = torch.device("cpu")
device=torch.device('cuda')
kwargs = {}



##############数据准备
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
#功能：加载保存到path中的各层参数到神经网络
#path='dataset/qiuguan/model_new_K_fold_RandomTree/MLP_non_encode/MLP9110.pkl'
#path='dataset/qiuguan/origin_800/non_encode_aug/para0.4_0.6_0/MLP10210.pkl'
path='dataset/qiuguan/origin_800/non_encode_aug/aug_with_simple/para_gause_0.8_0.6/MLP10710.pkl'

#path='dataset/qiuguan/origin_800/non_encode_aug/aug_model/para_gause_0.4_0.2_0.6_0.2/MLP7910.pkl'
#path='dataset/qiuguan/origin_800/non_encode_aug/aug_et/para_gause_0.8_0.5_gause_0.3_0.1/MLP11012345.pkl'
#nfm=NFM(nfm_config)
mlp=MLP()
#print(nfm)
#net = nn.DataParallel(net)
#net = net.to(device)
mlp.load_state_dict(torch.load(path),strict=False)
mlp.cuda()

print(mlp)




mlp_params = list(mlp.named_parameters())
#print(nfm_params)
net=mlp
model=net###########
'''
testset = KZDatasetTest(csv_path='../NFM-pyorch-master/dataset/qiuguan/orign/')
   
test_loader = DataLoader(
         dataset=testset,
         #transform=torchvision.transforms.ToTensor(),
         
         batch_size=nfm_config['batch_size']
        
     )
'''

testset_xiaoqiu  = KZDatasetTest(csv_path='dataset/qiuguan/origin_800/xiaoqiu/test_info.csv')#样本收集特征数据集，和测试数据集不同，这里边可能还包含训练集
   
test_loader_xiaoqiu = DataLoader(
         dataset=testset_xiaoqiu,
         #transform=torchvision.transforms.ToTensor(),
         
         batch_size=nfm_config['batch_size'],
         shuffle=True
        
     )

testset_xiaoguan  = KZDatasetTest(csv_path='dataset/qiuguan/origin_800/xiaoguan/test_info.csv')
   
test_loader_xiaoguan = DataLoader(
         dataset=testset_xiaoguan,
         #transform=torchvision.transforms.ToTensor(),
         
         batch_size=nfm_config['batch_size'],
         shuffle=True
        
     )

################小球

#LRP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#get_ipython().run_line_magic('matplotlib', 'inline')

from innvestigator import InnvestigateModel
#from utils import Flatten


inn_model = InnvestigateModel(model, lrp_exponent=2,
                              method="e-rule",
                              beta=.5)

genes_features=np.array([i for i  in range(9)])#################
genes_features=genes_features.reshape(9,1).tolist()#######################genes_features[i][0]=label



###[3, 25.353026075712233, tensor([ 182,  879,  103, 2657, 2489,  914, 2437,  180, 2417, 1402, 2344, 2947,
###values=tensor([0.0572, 0.0495, 0.0404, 0.0381, 0.0364, 0.0353, 0.0328, 0.0302, 0.0284,
#        0.0257, 0.0180, 0.0164, 0.0155, 0.0150, 0.0135, 0.0127, 0.0121, 0.0120,
#        0.0114, 0.0107, 0.0107, 0.0104, 0.0100, 0.0100, 0.0099, 0.0091, 0.0091,
#       0.0088, 0.0083, 0.0082, 0.0081, 0.0079, 0.0078, 0.0077, 0.0076, 0.0074,
#       0.0073, 0.0072, 0.0072, 0.0071], dtype=torch.float64),
###indices=tensor([ 182,  879,  103, 2657, 2489,  914, 2437,  180, 2417, 1402, 2344, 2947,
#        2546, 1114,  796, 1111, 2472, 2326, 1274,  932, 2476,  716,  989, 3289,
#        1252, 2053,  785, 2429, 3015, 1585,  975, 1150, 1155,  726,  823, 2303,
#         699,  349, 1792, 1524])), 25.359040192320702, tensor([2489,  826, 1753, 1792,  601, 2303, 1053,  545, 2559,  624, 3256,  762,
#        2666,  182, 1881, 1585,  726, 1367, 2405, 1171, 2947, 2093,   14,  265,
#         716,  180, 1467, 2207, 3223,  349,  277, 2141, 2878, 2427, 2326, 1111,
#         746, 1402, 2150, 1602]), torch.return_types.topk(
#values=tensor([0.0456, 0.0445, 0.0406, 0.0349, 0.0277, 0.0273, 0.0246, 0.0236, 0.0225,
#        0.0221, 0.0213, 0.0200, 0.0185, 0.0182, 0.0181, 0.0153, 0.0151, 0.0142,
##        0.0141, 0.0140, 0.0138, 0.0123, 0.0121, 0.0115, 0.0109, 0.0108, 0.0107,
#        0.0102, 0.0102, 0.0098, 0.0094, 0.0093, 0.0092, 0.0089, 0.0082, 0.0081,
#        0.0080, 0.0080, 0.0079, 0.0076], dtype=torch.float64),
#indices=tensor([2489,  826, 1753, 1792,  601, 2303, 1053,  545, 2559,  624, 3256,  762,
#        2666,  182, 1881, 1585,  726, 1367, 2405, 1171, 2947, 2093,   14,  265,
#         716,  180, 1467, 2207, 3223,  349,  277, 2141, 2878, 2427, 2326, 1111,
#         746, 1402, 2150, 1602]))]
model.double()
for data, target in test_loader_xiaoqiu:############小球

    data, target = data.to(device), target.to(device)
    #targets=torch.max(targets,1)[1]###################
    #print('data:',data.shape)
    batch_size = int(data.size()[0])
    #print('batch_size:',batch_size)#=20
    evidence_for_class = []
    #print("target:",target.shape)
    #print('target:',target[3])
    # Overlay with noise 
    # data[0] += 0.25 * data[0].max() * torch.Tensor(np.random.randn(28*28).reshape(1, 28, 28))
    #model_prediction, true_relevance = inn_model.innvestigate(in_tensor=data)

    for i in range(9):#10类
    # Unfortunately, we had some issue with freeing pytorch memory, therefore 
    # we need to reevaluate the model separately for every class.
        model_prediction, input_relevance_values = inn_model.innvestigate2(in_tensor=data, rel_for_class=i,target=target)
        evidence_for_class.append(input_relevance_values)
    print('input_relevance_values:',input_relevance_values.shape)
    print('evidence_for_class:',len(evidence_for_class))
    evidence_for_class = np.array([elt.numpy() for elt in evidence_for_class])
    print('evidence_for_class:',evidence_for_class.shape)#[10,20,784]
    
    #continue
    for idx, example in enumerate(data):#batch 中的每一个样本
        #print('example:',example.shape)
        prediction = np.argmax(model_prediction.cpu().detach(), axis=1)#
        print('prediction[idx]:',prediction[idx])
        print('evidence_for_class:',evidence_for_class[prediction[idx]][idx].shape)
        #fig, axes = plt.subplots(3, 5)
        '''
        fig.suptitle("Prediction of model: " + str(prediction[idx]) + "({0:.2f})".format(
            100*float(model_prediction[idx][model_prediction[idx].argmax()].exp()/model_prediction[idx].exp().sum())))
        '''
        prediction_value=prediction[idx]
        p_x=model_prediction[idx][model_prediction[idx].argmax()].exp()
        p_sum=model_prediction[idx].exp().sum()
        prediction_score=100*float(model_prediction[idx][model_prediction[idx].argmax()].exp()/model_prediction[idx].exp().sum())
        print('prediction_value:',prediction_value)
        print('prediction_score:',prediction_score)

        continue
        #print('分子:',p_x)
        #print('分母：',p_sum)
        #uu=pr
        #print("torch.argmax:",torch.argmax(target[idx]))
        if len(genes_features[prediction_value])==1:#有值，但还没有添加预测分数和特征值，只有标签#prediction_value代表第几种疾病
            if prediction_value!=torch.argmax(target[idx]).cpu().detach()  :
                print('不合格****************:',prediction_value)
            if prediction_value==torch.argmax(target[idx]).cpu().detach()  :#预测正确
                genes_features[prediction_value].append(prediction_score)
                print('合格：',prediction_value)
                relevance_score_for_every_pixel=evidence_for_class[prediction[idx]][idx]
                #print('relevance_score_for_every_pixel.shape:',relevance_score_for_every_pixel.tolist())
                relevance_score_for_every_pixel=torch.from_numpy(relevance_score_for_every_pixel)
                index=torch.topk(relevance_score_for_every_pixel,10,largest=True)#基因个数50#####150
                #print('pixel_sorted:',index)
                genes_features[prediction_value].append(index.indices)#添加前50基因特征
                genes_features[prediction_value].append(index)
        else: 
            if genes_features[prediction_value][1]<prediction_score:#如果值比已有值大，说明预测更准确
                if prediction_value==torch.argmax(target[idx]).cpu().detach():
                    print('&&&&&&&&&&&&&&&&&&&合   格&&&&&&&&&&&&&&:',prediction_value)
                    genes_features[prediction_value].pop(3)#删除index
                    genes_features[prediction_value].pop(2)#先删除特征值
                    genes_features[prediction_value].pop(1)#先删除预测分数
                
                    genes_features[prediction_value].append(prediction_score)
        
                    relevance_score_for_every_pixel=evidence_for_class[prediction[idx]][idx]
                    #print('relevance_score_for_every_pixel.shape:',relevance_score_for_every_pixel.tolist())
                    relevance_score_for_every_pixel=torch.from_numpy(relevance_score_for_every_pixel)
                    index=torch.topk(relevance_score_for_every_pixel,10,largest=True)#基因个数50  200
                    #print('pixel_sorted:',index)
                    genes_features[prediction_value].append(index.indices)
                    genes_features[prediction_value].append(index)
print('qiu_____genes_features.shape:',genes_features)#######找特征基因#############从3301中找200个基因








