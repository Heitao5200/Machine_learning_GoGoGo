#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : CV1.py
# @Date    : 2018-11-22
# @Author  : 黑桃
# @Software: PyCharm 
from pandas import Series, DataFrame
import pickle
from sklearn import svm
from sklearn.model_selection import *	#划分数据 交叉验证
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")
path = "E:/MyPython/Machine_learning_GoGoGo/"
"""=====================================================================================================================
1 读取数据
"""
print("0 读取特征")
f = open(path + 'feature/feature_V3.pkl', 'rb')
train, test, y_train,y_test = pickle.load(f)
f.close()

"""=====================================================================================================================
2 进行K次训练；用K个模型分别对测试集进行预测，并得到K个结果，再进行结果的融合
"""
preds = []
i = 0

"""=====================================================================================================================
3  交叉验证方式
"""
## 对交叉验证方式进行指定，如验证次数，训练集测试集划分比例等
kf = KFold(n_splits=5, random_state=1)
loo = LeaveOneOut()#将数据集分成训练集和测试集，测试集包含一个样本，训练集包含n-1个样本
lpo = LeavePOut(p=2000)## #将数据集分成训练集和测试集，测试集包含p个样本，训练集包含n-p个样本
ss= ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
tss = TimeSeriesSplit(n_splits=5)

logo = LeaveOneGroupOut()
lpgo = LeavePGroupsOut(n_groups=3)
gss = GroupShuffleSplit(n_splits=4, test_size=.5, random_state=0)
gkf = GroupKFold(n_splits=2)
"""【配置交叉验证方式】"""
cv=ss

clf = svm.SVC(kernel='linear', C=1)
score_sum = 0
# 原始数据的索引不是从0开始的，因此重置索引
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
for train_idx, vali_idx in cv.split(train, y_train):
    i += 1
    """获取训练集和验证集"""
    f_train_x = DataFrame(train[train_idx])
    f_train_y = DataFrame(y_train[train_idx])
    f_vali_x = DataFrame(train[vali_idx])
    f_vali_y = DataFrame(y_train[vali_idx])

    """训练分类器"""
    classifier = svm.LinearSVC()
    classifier.fit(f_train_x, f_train_y)

    """对测试集进行预测"""
    y_test = classifier.predict(test)
    preds.append(y_test)

    """对验证集进行预测，并计算f1分数"""
    pre_vali = classifier.predict(f_vali_x)
    score_vali = f1_score(y_true=f_vali_y, y_pred=pre_vali, average='macro')
    print("第{}折， 验证集分数：{}".format(i, score_vali))
    score_sum += score_vali
    score_mean = score_sum / i
    print("第{}折后， 验证集分平均分数：{}".format(i, score_mean))





