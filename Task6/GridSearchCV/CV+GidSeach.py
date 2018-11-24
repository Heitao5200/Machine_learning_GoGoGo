#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : CV+GidSeach.py
# @Date    : 2018-11-23
# @Author  : 黑桃
# @Software: PyCharm
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import *
import warnings
warnings.filterwarnings("ignore")
path = "E:/MyPython/Machine_learning_GoGoGo/"
"""=====================================================================================================================
1 读取数据
"""
print("0 读取特征")
f = open(path + 'feature/feature_V3.pkl', 'rb')
train_X,test_X,train_y,test_y = pickle.load(f)
f.close()

X_trainval,X_test,y_trainval,y_test = train_X,test_X,train_y,test_y
X_train,X_val,y_train,y_val = train_test_split(train_X,train_y,random_state=1)
print("Size of training set:{} size of validation set:{} size of teseting set:{}".format(X_train.shape[0],X_val.shape[0],X_test.shape[0]))

"""
3.1 交叉验证方式
"""
## 对交叉验证方式进行指定，如验证次数，训练集测试集划分比例等
kf = KFold(n_splits=3, random_state=1)
loo = LeaveOneOut()#将数据集分成训练集和测试集，测试集包含一个样本，训练集包含n-1个样本
lpo = LeavePOut(p=2000)## #将数据集分成训练集和测试集，测试集包含p个样本，训练集包含n-p个样本
ss= ShuffleSplit(n_splits=3, test_size=.25, random_state=0)
tss = TimeSeriesSplit(n_splits=3)

logo = LeaveOneGroupOut()
lpgo = LeavePGroupsOut(n_groups=3)
gss = GroupShuffleSplit(n_splits=4, test_size=.5, random_state=0)
gkf = GroupKFold(n_splits=2)

cv=kf

best_score = 0.0
for gamma in [0.001,0.01,0.1,1,10,100]:
    for C in [0.001,0.01,0.1,1,10,100]:
        svm = SVC(gamma=gamma,C=C)
        scores = cross_val_score(svm,X_trainval,y_trainval,cv=cv) #5折交叉验证
        score = scores.mean() #取平均数
        print("当前gamma值:{} ,    当前C值:{},    当前分数:{}".format(gamma, C, score))
        if score > best_score:
            best_score = score
            best_parameters = {"gamma":gamma,"C":C}
svm = SVC(**best_parameters)
svm.fit(X_trainval,y_trainval)
test_score = svm.score(X_test,y_test)
print("Best score on validation set:{:.2f}".format(best_score))
print("Best parameters:{}".format(best_parameters))
print("Score on testing set:{:.2f}".format(test_score))

