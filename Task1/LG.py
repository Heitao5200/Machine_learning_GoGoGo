#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : LG.py
# @Date    : 2018-11-14
# @Author  : 黑桃
# @整理    : 等到的过去


import pickle
import pandas as pd #数据分析
from pandas import Series,DataFrame
from sklearn.model_selection import train_test_split
import time
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
print("开始......")
t_start = time.time()
path = "E:/MyPython/Machine_learning_GoGoGo/"
"""=====================================================================================================================
1 读取特征
"""
print("0 读取原特征")
f = open(path + 'feature/feature_V3.pkl', 'rb')
train, test, y_train,y_test = pickle.load(f)
f.close()
"""=====================================================================================================================
5 模型训练
"""
print("模型训练")

lg = LogisticRegression(C=10,dual=True)
lg.fit(train,y_train)

"""【保存模型】"""
print('3 保存模型')
joblib.dump(lg, path + "model/model_file/lg_120.pkl")


"""=====================================================================================================================
6 模型预测
"""
y_test_pre = lg.predict(test)

"""=====================================================================================================================
7 模型评分
"""

print("验证集分数：{}".format(f1_score(y_test, y_test_pre, average='macro')))


print("_Train_AUC Score ：{:.4f}".format(lg.score(train, y_train)))
print("_Train_AUC Score ：{:.4f}".format(lg.score(test, y_test)))

