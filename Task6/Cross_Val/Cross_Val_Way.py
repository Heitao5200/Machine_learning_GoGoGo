#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : Cross_Val_Way.py
# @Date    : 2018-11-23
# @Author  : 黑桃
# @Software: PyCharm 

from pandas import Series, DataFrame
import pickle
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn import datasets	#自带数据集

from sklearn.neighbors import KNeighborsClassifier  #一个简单的模型，只有K一个参数，类似K-means
import matplotlib.pyplot as plt
from sklearn import metrics

path = "E:/MyPython/Machine_learning_GoGoGo/"
"""=====================================================================================================================
1 读取数据
"""
print("0 读取特征")
f = open(path + 'feature/feature_V3.pkl', 'rb')
train_X,test_X,train_y,test_y = pickle.load(f)
f.close()
clf = svm.SVC(kernel='linear', C=1)
"""
cross_val_score默认是以 scoring=’f1_macro’
"""
from sklearn.model_selection import cross_val_score
## 使用默认交叉验证方式
scores = cross_val_score(clf,train_X, train_y, cv=5)
print(scores)

## 对交叉验证方式进行指定，如验证次数，训练集测试集划分比例等
cv = ShuffleSplit(n_splits=5, test_size=.3, random_state=0)
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
predicted = cross_val_predict(clf, train_X, train_y, cv=5)
predicted_score=metrics.accuracy_score(train_y, predicted)
print(predicted_score)

""" 
KFold
"""
#K折交叉验证，这是将数据集分成K份的官方给定方案，所谓K折就是将数据集通过K次分割
# 使得所有数据既在训练集出现过，又在测试集出现过，当然，每次分割中不会有重叠。相当于无放回抽样。
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)#将数据集分成2份的官方给定方案
X = np.ones(12)
y = [0,0,0,0,1,1,1,1,1,1,1,1]
# scores = KFold(train_X, train_y)
print("----------------KFold----------------")
for train, test in kf.split(X,y):
    print(train, test)
    # print(np.array(train_y)[train], np.array(train_y)[test])


"""
LeaveOneOut
"""
## LeaveOneOut 其实就是KFold 的一个特例，因为使用次数比较多，因此独立的定义出来，完全可以通过KFold 实现。
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()#将数据集分成训练集和测试集，测试集包含一个样本，训练集包含n-1个样本
print("----------------LeaveOneOut----------------")
for train, test in loo.split(train_y):
    print(train, test)
"""
LeavePOut
"""
# #这个也是KFold 的一个特例，用KFold 实现起来稍麻烦些，跟LeaveOneOut 也很像
from sklearn.model_selection import LeavePOut
lpo = LeavePOut(p=3)## #将数据集分成训练集和测试集，测试集包含p个样本，训练集包含n-p个样本
print("----------------LeavePOut----------------")
for train, test in lpo.split(train_y):
    print(train, test)

"""
ShuffleSplit
"""
## ShuffleSplit 咋一看用法跟LeavePOut 很像，其实两者完全不一样，
# LeavePOut 是使得数据集经过数次分割后，所有的测试集出现的元素的集合即是完整的数据集，
# 即无放回的抽样，而ShuffleSplit 则是有放回的抽样，只能说经过一个足够大的抽样次数后，
# 保证测试集出现了完成的数据集的倍数。
from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=3, test_size=.25, random_state=0)
print("----------------ShuffleSplit----------------")
for train, test in ss.split(train_y):
    print(train, test)

"""
StratifiedKFold
"""
#通过指定分组，对测试集进行无放回抽样。【指定分组具体是怎么指定的？？？？】
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3)
X = np.ones(12)
y = [0,0,0,0,1,1,1,1,1,1,1,1]
print("----------------StratifiedKFold----------------")

for train, test in skf.split(X,y):
    print(train, test)


"""
GroupKFold
"""
# 这个跟StratifiedKFold 比较像，不过测试集是按照一定分组进行打乱的，
# 即先分堆，然后把这些堆打乱，每个堆里的顺序还是固定不变的。
from sklearn.model_selection import GroupKFold
X = [.1, .2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ['a','b','b','b','c','c','c','d','d','d']
groups = [1,1,1,2,2,2,3,3,3,3]
print("----------------GroupKFold----------------")
gkf = GroupKFold(n_splits=2)
for train, test in gkf.split(X,y,groups=groups):
    print(train, test)

"""
LeaveOneGroupOut
"""
# 这个是在GroupKFold 上的基础上混乱度又减小了，按照给定的分组方式将测试集分割下来。
from sklearn.model_selection import LeaveOneGroupOut
X = [1, 5, 10, 50, 60, 70, 80]
y = [0, 1, 1, 2, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3, 3]
logo = LeaveOneGroupOut()
print("----------------LeaveOneGroupOut----------------")
for train, test in logo.split(X, y, groups=groups):
    print(train, test)

"""
LeavePGroupsOut
"""
# 跟上面那个一样，只是一个是单组，一个是多组
from sklearn.model_selection import LeavePGroupsOut
X = np.arange(6)
y = [1, 1, 1, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3]
lpgo = LeavePGroupsOut(n_groups=2)
print("----------------LeavePGroupsOut----------------")
for train, test in lpgo.split(X, y, groups=groups):
    print(train, test)
"""
GroupShuffleSplit
"""
# 这个是有放回抽样
from sklearn.model_selection import GroupShuffleSplit
X = [.1, .2, 2.2, 2.4, 2.3, 4.55, 5.8, .001]
y = ['a', 'b','b', 'b', 'c','c', 'c', 'a']
groups = [1, 1, 2, 2, 3, 3, 4, 4]
print("----------------GroupShuffleSplit----------------")
gss = GroupShuffleSplit(n_splits=4, test_size=.5, random_state=0)
for train, test in gss.split(X, y, groups=groups):
    print(train, test)

"""
TimeSeriesSplit
"""
# 针对时间序列的处理，防止未来数据的使用，分割时是将数据进行从前到后切割（这个说法其实不太恰当，因为切割是延续性的。。）
from sklearn.model_selection import TimeSeriesSplit
X = np.array([[1,2],[3,4],[1,2],[3,4],[1,2],[3,4]])
tscv = TimeSeriesSplit(n_splits=3)
print("----------------TimeSeriesSplit----------------")
for train, test in tscv.split(X):
    print(train, test)














