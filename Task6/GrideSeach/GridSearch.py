#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : GridSearch.py
# @Date    : 2018-11-23
# @Author  : 黑桃
# @Software: PyCharm
import pickle
from sklearn.svm import SVC

path = "E:/MyPython/Machine_learning_GoGoGo/"
"""=====================================================================================================================
1 读取数据
"""
print("0 读取特征")
f = open(path + 'feature/feature_V3.pkl', 'rb')
X_train,X_test,y_train,y_test = pickle.load(f)
f.close()

print("训练集:{}  测试集:{}".format(X_train.shape[0],X_test.shape[0]))
"""=====================================================================================================================
2 网格搜索
"""
# X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.3,random_state=1)
X_train,X_val,y_train,y_val = X_train,X_test,y_train,y_test
print("训练集:{} 验证集:{} 测试集:{}".format(X_train.shape[0],X_val.shape[0],X_test.shape[0]))
best_score = 0
for gamma in [0.001,0.01,0.1,1,10,100]:
    for C in [0.001,0.01,0.1,1,10,100]:
        svm = SVC(gamma=gamma,C=C)
        svm.fit(X_train,y_train)
        score = svm.score(X_val,y_val)
        print("当前gamma值:{} ,    当前C值:{},    当前分数:{}".format(gamma, C, score))
        if score > best_score:
            best_score = score
            best_parameters = {'gamma':gamma,'C':C}
svm = SVC(**best_parameters) #使用最佳参数，构建新的模型
svm.fit(X_train,y_train) #使用训练集和验证集进行训练，more data always results in good performance.
test_score = svm.score(X_test,y_test) # evaluation模型评估
print("Best score on validation set:{}".format(best_score))
print("Best parameters:{}".format(best_parameters))
print("Best score on test set:{}".format(test_score))

