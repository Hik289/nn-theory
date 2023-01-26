import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import lightgbm as lgb
from sklearn.kernel_ridge import KernelRidge

from multiprocessing import Lock, Manager
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import time
import os
import sys

def generate_y(x,beta):
    # x is numpy array with (n,d) ,beta is (d) and fixed
    N = x.shape[0]
    d = x.shape[1]
    e = np.random.normal(0, 0.5,size = N)
    vector = np.einsum('i,ij -> j', beta,x.T)
    fx = np.sqrt(4/10)*vector + np.sqrt(4/10)*(np.sqrt(1/2)*(vector**2-1))+ \
        np.sqrt(2/10)*(np.sqrt(1/10)*(vector**4 - 6*vector**2 + 3))
    y = fx + e
    return y

def getRandomSamplesOnNSphere(N , numberOfSamples, R = 1):
    X = np.random.default_rng().normal(size=(numberOfSamples , N))
    return R / np.sqrt(np.sum(X**2, 1, keepdims=True)) * X

class dataset(Dataset):
    def __init__(self, data_tensor, data_target):
        self.data_target = data_target
        self.data_tensor = data_tensor 
    
    def __len__(self):
        return self.data_target.shape[0]

    def __getitem__(self, index):
         return self.data_tensor[index], self.data_target[index]   


class Model_1(nn.Module):
    def __init__(self, input_dim, Nd = 100, drop_rate = 0.0, ):
        super(Model_1,self).__init__()
        self.model_name = '2 layer linear nn'
        
        self.hidden_dim = Nd//input_dim
        
        self.linear1 = nn.Linear(input_dim,self.hidden_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_dim,1)

        # torch.set_num_threads(1)

#         for p in self.linear1.parameters():
#             nn.init.normal_(p,mean=0.0,std = 0.001)
#         for p in self.linear2.parameters():
#             nn.init.normal_(p,mean=0.0,std = 0.001)


    def forward(self,x):

        x_signal = self.linear1(x)
        x_signal = self.act(x_signal)
        out = self.linear2(x_signal)

        return out


if __name__ == '__main__':
    nrep = 10
    d = 20
    beta = np.array([ 0.64653496,  0.60943061, -0.66344218,  0.03155915,  0.43562872,
       -1.59973928, -0.2902682 , -1.61472201, -0.18811938, -0.8680576 ,
       -1.11660674, -1.55543837,  2.1161863 , -1.117913  ,  0.22876578,
        0.88415557, -0.40531932, -0.48617956,  0.9729298 ,  1.09236064])
    Nd = int(sys.argv[1])
    result_trainingerror = np.zeros((1,40))
    result_testerror = np.zeros((1,40))    

    x_id = Nd /20
    for j in range(20,60):
        y_id = j/20
        
        N = int(np.ceil(np.e**(y_id * np.log(20))))
        Nd = int(np.ceil(np.e**(x_id* np.log(20))))
        
        for nrepitition in range(nrep):
            x = getRandomSamplesOnNSphere(d,N)
            y = generate_y(x,beta)

            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,shuffle = True)
            x_train = torch.FloatTensor(x_train)
            x_test = torch.FloatTensor(x_test)
            y_train = torch.FloatTensor(y_train).unsqueeze(1)
            y_test = torch.FloatTensor(y_test).unsqueeze(1)

            train_dataset = dataset(x_train, y_train)
            test_dataset = dataset(x_test, y_test)
            train_dataloader = DataLoader(dataset= train_dataset, 
                                        batch_size = len(x_train), 
                                        shuffle= True, 
                                        drop_last= False)
            test_dataloader = DataLoader(dataset= test_dataset, 
                                        batch_size = len(x_test), 
                                        shuffle= True, 
                                        drop_last= False)
            torch.cuda.empty_cache()

            lr = 0.01

            device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
            model = Model_1(input_dim= d, Nd = Nd,drop_rate= 0.0).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr)
            LOSS = 0
            LOSS2 = 0
            model.train()
            
            for epoch in range(500):

                for index, (x, y) in enumerate(train_dataloader):
                    if torch.cuda.is_available():
                        x = x.to(device)
                        y = y.to(device)
                    y_pred = model(x)
                    loss = criterion(y_pred,y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            #         for p in model.parameters():
            #             # print(p.grad.norm())                 
            #             torch.nn.utils.clip_grad_norm_(p, 10)  
            #         optimizer.step()
            LOSS += loss
            model.eval()
            loss2 = criterion(model(x_test), y_test)
            
            LOSS2 += loss2
            
        training_error = LOSS /nrep
        test_error = LOSS2 /nrep
        
        result_trainingerror[0][j-20] = training_error
        result_testerror[0][j-20] = test_error
        
        print('Nd: %d, N: %d,trainingerror: %f, testerror: %f'%(Nd,N, training_error, test_error))
        
    np.save('./result/%d training_error.npy'%Nd,result_trainingerror)
    np.save('./result/%d test_error.npy'%Nd,result_testerror)         