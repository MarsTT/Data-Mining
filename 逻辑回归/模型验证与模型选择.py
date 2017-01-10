# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:49:31 2017

@author: Mars
"""
#构建数据集
from sklearn import linear_model,metrics
from sklearn.cross_validation import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

model_data = pd.read_csv("date_data.csv")
#model_data.head()
Y = model_data["Dated"]
X = model_data.ix[:,'income':'assets']
train_data,test_data,train_target,test_target = train_test_split(X,Y,test_size = 0.2,random_state=0)

#建模
logistic_model = linear_model.LogisticRegression()
logistic_model.fit(train_data,train_target)

test_est = logistic_model.predict(test_data)
train_est = logistic_model.predict(train_data)
test_est_p = logistic_model.predict_proba(test_data)[:,1]
train_est_p = logistic_model.predict_proba(train_data)[:,1]

#决策类检验
print metrics.classification_report(test_target,test_est)

metrics.accuracy_score(test_target,test_est)


#排序类检验
#ROC曲线
fpr_test,tpr_test,th_test = metrics.roc_curve(test_target,test_est_p)
fpr_train,tpr_train,th_train = metrics.roc_curve(train_target,train_est_p)
plt.figure(figsize=[6,6])
plt.plot(fpr_test,tpr_test,color = 'red')
plt.plot(fpr_train,tpr_train,color = 'black')
plt.plot()
plt.title('ROC curve')

test_AUC = metrics.roc_auc_score(test_target,test_est_p)
train_AUC = metrics.roc_auc_score(train_target,train_est_p)
print("test_AUC:",test_AUC,"train_AUC:",train_AUC)

#KS曲线
test_x_axis = np.arange(len(fpr_test))/float(len(fpr_test))
train_x_axis = np.arange(len(fpr_train))/float(len(fpr_train))
plt.figure(figsize=[6,6])
plt.plot(fpr_test,test_x_axis,color = 'blue')
plt.plot(fpr_train,train_x_axis,color = 'red')
plt.title('KS curve')


















