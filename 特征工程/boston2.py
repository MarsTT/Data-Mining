# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 02:46:22 2017

@author: beta
"""
#linalg计算
from sklearn.datasets import load_boston
boston=load_boston()
import numpy as np

x=boston.data[:,5]
x=np.array([[v,1] for v in x])

y=boston.target
(slope,bias),total_error,_,_ = np.linalg.lstsq(x,y)
rmse=np.sqrt(total_error[0]/len(x))  #均方根误差
print rmse

#最小二乘法计算
from sklearn.linear_model import LinearRegression
y=boston.target
lr=LinearRegression(fit_intercept=True)
lr.fit(x,y)

p=lr.predict(x)
e=p-y

total_error=np.sum(e*e)
rmse_train=np.sqrt(total_error/len(p))
print 'RMSE on training:{}'.format(rmse_train) 

#交叉验证计算
from sklearn import cross_validation   
kfold = cross_validation.KFold(len(x), n_folds=10)
err=0
for train,test in kfold:
    lr.fit(x[train],y[train])
    p=lr.predict(x[test])
    e=p-y[test]
    err+=np.sum(e*e)
rmse_10cv=np.sqrt(err/len(x))
print 'RMSE on 10-fold CV:{}'.format(rmse_10cv)

