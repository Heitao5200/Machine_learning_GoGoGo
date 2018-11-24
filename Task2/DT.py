#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : DT.py
# @Date    : 2018-11-16
# @Author  : 黑桃
# @Software: PyCharm 



import pickle
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,r2_score
from sklearn.externals import joblib
print("开始......")
t_start = time.time()
path = "E:/MyPython/Machine_learning_GoGoGo/"
"""=====================================================================================================================
1 读取特征
"""
print("0 读取特征")
f = open(path + 'feature/feature_V3.pkl', 'rb')
train, test, y_train,y_test = pickle.load(f)
f.close()
"""=====================================================================================================================
2 模型训练
"""
print("决策树模型训练")
DT = DecisionTreeClassifier(max_depth=4)
DT.fit(train,y_train)

"""【保存模型】"""
print('3 保存模型')
joblib.dump(DT, path + "model/model_file/DT.pkl")
"""=====================================================================================================================
3 模型预测
"""
y_test_pre = DT.predict(test)

"""=====================================================================================================================
4 决策树模型评分
"""

print("f1分数：{}".format(f1_score(y_test, y_test_pre, average='macro')))
# r2 = r2_score(y_test, y_test_pre)## R2评分多用于回归
# print("f2分数：{}".format(r2))

print("验证集分数：{}".format( DT.score(test, y_test)))

