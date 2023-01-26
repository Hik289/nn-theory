from tasks import task
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
if __name__ == '__main__':
    pool = Pool(8, maxtasksperchild=1)
    training_error_result = np.empty((0,40))
    test_error_result = np.empty((0,40))
    res1 = []
    nrep = 10
    d = 20
    beta = np.array([ 0.64653496,  0.60943061, -0.66344218,  0.03155915,  0.43562872,
       -1.59973928, -0.2902682 , -1.61472201, -0.18811938, -0.8680576 ,
       -1.11660674, -1.55543837,  2.1161863 , -1.117913  ,  0.22876578,
        0.88415557, -0.40531932, -0.48617956,  0.9729298 ,  1.09236064])
    for i in range(20,80):
        res = pool.apply_async(task,args = (i,nrep,beta,d,))
        res1.append(res.get())
   
    pool.close()
    pool.join()
    for res in res1:
        training_error_result = np.vstack(training_error_result,res[1])
        test_error_result = np.vstack(test_error_result, res[2])
    
    np.save('./training_error_result.npy',training_error_result)
    np.save('./test_error_result.npy',test_error_result)   