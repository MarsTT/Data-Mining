# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 21:35:04 2017

@author: beta
"""

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

xArr,yArr=loadDataSet('ex0.txt')

def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

ws=standRegres(xArr,yArr)
xMat=mat(xArr)
yMat=mat(yArr) 
yHat=xMat*ws

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])

xCopy=xMat.copy()
xCopy.sort(0)
yHat=xCopy * ws

ax.plot(xCopy[:,1],yHat)
plt.show()

yHat=xMat*ws
print corrcoef(yHat.T,yMat)


