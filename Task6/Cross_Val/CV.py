#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : CV.py
# @Date    : 2018-11-23
# @Author  : 黑桃
# @Software: PyCharm 

import warnings
warnings.filterwarnings("ignore")
from pandas import Series, DataFrame
import pickle

from sklearn import svm
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import *

from sklearn import metrics

path = "E:/MyPython/Machine_learning_GoGoGo/"
"""=====================================================================================================================
1 读取数据
"""
print("0 读取特征")
f = open(path + 'feature/feature_V3.pkl', 'rb')
train_X,test_X,train_y,test_y = pickle.load(f)
f.close()

"""=====================================================================================================================
2 模型参数设置
"""
SVM_linear = svm.SVC(kernel='linear', C=1)
SVM_poly = svm.SVC(kernel='poly', C=1)

clf = SVM_linear

"""
3 交叉验证评分：cross_val_score默认是以 scoring=’f1_macro’
"""
scores = cross_val_score(clf,train_X, train_y, cv=5)
print(scores)
print(scores.mean())
"""
3.1 交叉验证方式
"""
## 对交叉验证方式进行指定，如验证次数，训练集测试集划分比例等
kf = KFold(n_splits=3, random_state=1)
loo = LeaveOneOut()#将数据集分成训练集和测试集，测试集包含一个样本，训练集包含n-1个样本
lpo = LeavePOut(p=2000)## #将数据集分成训练集和测试集，测试集包含p个样本，训练集包含n-p个样本
ss= ShuffleSplit(n_splits=3, test_size=.25, random_state=0)
tss = TimeSeriesSplit(n_splits=3)
##
logo = LeaveOneGroupOut()
lpgo = LeavePGroupsOut(n_groups=3)
gss = GroupShuffleSplit(n_splits=4, test_size=.5, random_state=0)
gkf = GroupKFold(n_splits=2)

cv=ss

scores = cross_val_score(clf, train_X, train_y, cv=cv)
print(scores)

## 在cross_val_score 中同样可使用pipeline 进行流水线操作
clf = make_pipeline(preprocessing.StandardScaler(), clf)
scores2 = cross_val_score(clf, train_X, train_y, cv=cv)
print(scores2)

"""
cross_val_predict
"""
## cross_val_predict 与cross_val_score 很相像，
## 不过不同于返回的是评测效果，cross_val_predict 返回的是estimator 的分类结果（或回归值）
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(clf, train_X, train_y, cv=5)##cross_val_predict only works for partitions
predicted_score=metrics.accuracy_score(train_y, predicted)
print(predicted_score)