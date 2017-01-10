# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:40:42 2017

@author: Mars
"""

'''
    数据说明：本数据是一份汽车贷款违约数据
    
    application_id   申请者ID
    account_number   账户号
    bad_ind          是否违约
    vehicle_year     汽车购买时间
    vehicle_make     汽车制造商
    bankruptcy_ind   曾经破产标识
    tot_derog        五年内信用不良事件数量(比如手机欠费销号)
    tot_tr           全部账户数量
    age_oldest_tr    最久账号存续时间(月)
    tot_open_tr      在使用账户数量
    tot_rev_tr       在使用可循环贷款账户数量(比如信用卡)
    tot_rev_debt     在使用可循环贷款账户余额(比如信用款欠款)
    tot_rev_line     可循环贷款账户限额(信用卡授权额度)
    rev_util         可循环贷款账户使用比列(余额/限额)
    fico_socre       FICO打分
    purch_price      汽车购买金额(元)
    msrp             建议售价
    down_pyt         分期付款的首次交款
    loan_term        贷款期限(月)
    loan_amt         贷款金额
    ltv              贷款金额/建议售价*100
    tot_income       月均收入(元)
    veh_mileage      行使历程(Mile)
    uesd_ind         是否二手车
    weight           样本权重
'''

import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#导入数据并进行清洗
accepts = pd.read_csv('accepts.csv',skipinitialspace=True)
accepts = accepts.dropna(axis=0,how='any')

#1.分类变量的相关关系
#1.1曾经破产标识与是否违约是否有关系？
#交叉表
cross_table = pd.crosstab(accepts.bankruptcy_ind,accepts.bad_ind,margins=False)
crossTable = pd.crosstab(accepts.bankruptcy_ind,accepts.bad_ind,margins=True)

#列联表
def percConvert(ser):
    return ser/float(ser[-1])
crossTable.apply(percConvert,axis=0)

print 'chisp = %6.4f\n p-value = %6.4f\n dof=%i\n excepted_freq = %s'\
         %stats.chi2_contingency(cross_table)
         
#逻辑回归     
accepts.plot(x='fico_score',y='bad_ind',kind='scatter')

#Generalized linear model regression results
lg = smf.glm('bad_ind~fico_score',data=accepts,family=sm.families.Binomial(\
            sm.families.links.logit)).fit()
lg.summary()

accepts.describe(include='all')

#Generalized linear model regression results
formula = 'bad_ind~fico_score+C(bankruptcy_ind)+tot_derog+age_oldest_tr+\
            rev_uitl+ltv+veh_mileage'
lg = smf.glm(formula = formula,data=accepts,family=sm.families.Binomial(\
            sm.families.links.logit)).fit()
lg.summary()

#进行预测
accepts['p'] = lg.predict(accepts)
accepts.p.head()

#正则化表达
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_exog = np.array(accepts[['fico_score','tot_derog','age_oldest_tr','rev_util',\
                             'ltv','veh_mileage']])
for i in range(6):
    print variance_inflation_factor(vif_exog,i)
#结果显示第1，5个变量有共性问题


#python中没有现成的实现逐步回归法的包，因此使用sklearn中的正则化方法解决共线性问题

from datetime import datetime
import numpy as np
#import matplotlib.pyplot as plt

from sklearn import linear_model
#from sklearn import datasets
from sklearn.svm import l1_min_c

X = np.array(accepts[['fico_score','tot_derog','age_oldest_tr','rev_util',\
                      'ltv','veh_mileage']])
y = np.array(accepts['bad_ind'])

cs = l1_min_c(X,y,loss='log')*np.logspace(0,5)
print 'Computing regularization path....'
start = datetime.now()
clf = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
coef_ = []
for c in cs:
    clf.set_params(C=c)
    clf.fit(X,y)
    coef_.append(clf.coef_.ravel().copy())
print("this took:",datetime.now() - start)


coef_ = np.array(coef_)
plt.plot(np.log10(cs),coef_)
ymin,ymax = plt.ylim()
plt.xlabel('log(C)')
plt.ylabel('Coefficients')
plt.title('Logistic Regression Path')
plt.axis('tight')
plt.show()










