# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a python script file.
"""

import sklearn.datasets 
import sklearn.linear_model 
import numpy.random 
import numpy.linalg 
import matplotlib.pyplot 

if __name__ == "__main__": 
# 载入数据
  boston = sklearn.datasets.load_boston() 

# 使用样本比，拆分数据 
  sampleRatio = 0.5 
  n_samples = len(boston.target) 
  sampleBoundary = int(n_samples * sampleRatio) 

# 打乱全部数据
  shuffleIdx = range(n_samples) 
  numpy.random.shuffle(shuffleIdx) 


# 生成训练数据 
  train_features = boston.data[shuffleIdx[:sampleBoundary]] 
  train_targets = boston.target[shuffleIdx [:sampleBoundary]] 

# 生成测试数据
  test_features = boston.data[shuffleIdx[sampleBoundary:]] 
  test_targets = boston.target[shuffleIdx[sampleBoundary:]] 

# 训练
  linearRegression = sklearn.linear_model.LinearRegression() 
  linearRegression.fit(train_features, train_targets) 

# 预测
  predict_targets = linearRegression.predict(test_features) 

# 计算均方根误差
  n_test_samples = len(test_targets) 
  X = range(n_test_samples)
  error = numpy.linalg.norm(predict_targets - test_targets, ord = 1) / n_test_samples 
  print "Ordinary Least Squares (Boston) Error: %.2f" %(error)

# 画图
  matplotlib.pyplot.plot(X, predict_targets, 'r--', label = 'Predict Price')
  matplotlib.pyplot.plot(X, test_targets, 'g:', label='True Price')
  legend = matplotlib.pyplot.legend()
  matplotlib.pyplot.title("Ordinary Least Squares (Boston)")
  matplotlib.pyplot.ylabel("Price")
  matplotlib.pyplot.savefig("Ordinary Least Squares (Boston).png", format='png')
  matplotlib.pyplot.show()
