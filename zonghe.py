#!/user/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2018/11/20 18:39
# @Author  : 刘
# @Site    : 
# @File    : zonghe.py
# @Software: PyCharm
import pickle
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from  sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, recall_score,accuracy_score
from sklearn.metrics import auc,roc_auc_score,roc_curve,precision_score
import matplotlib.pyplot as plt

"""
1.读取特征
"""
path = "E:/MyPython/Machine_learning_GoGoGo/"
f = open(path + 'feature/feature_V1.pkl', 'rb')
train,test,y_train,y_test = pickle.load(f)
f.close()

# Lr
log_reg = LogisticRegression()
log_reg.fit(train, y_train)
# SVM
LinearSVC = svm.SVC(kernel='linear', probability=True).fit(train, y_train)
# decision tree
dtree = DecisionTreeClassifier(max_depth=6)
dtree.fit(train, y_train)
# xgboost
xgbClassifier = XGBClassifier()
xgbClassifier.fit(train, y_train)
# lightgbm
lgbmClassifier = LGBMClassifier()
lgbmClassifier.fit(train, y_train)

def model_metrics(clf, X_train, X_test, y_train, y_test):
    # 预测
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    # 准确率
    print('[准确率]', end=' ')
    print('训练集：', '%.4f' % accuracy_score(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % accuracy_score(y_test, y_test_pred))

    # 精准率
    print('[精准率]', end=' ')
    print('训练集：', '%.4f' % precision_score(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % precision_score(y_test, y_test_pred))

    # 召回率
    print('[召回率]', end=' ')
    print('训练集：', '%.4f' % recall_score(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % recall_score(y_test, y_test_pred))

    # f1-score
    print('[f1-score]', end=' ')
    print('训练集：', '%.4f' % f1_score(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % f1_score(y_test, y_test_pred))

    # auc取值：用roc_auc_score或auc
    print('[auc值]', end=' ')
    print('训练集：', '%.4f' % roc_auc_score(y_train, y_train_proba), end=' ')
    print('测试集：', '%.4f' % roc_auc_score(y_test, y_test_proba))

    # roc曲线
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_proba, pos_label=1)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_proba, pos_label=1)

    label = ["Train - AUC:{:.4f}".format(auc(fpr_train, tpr_train)),
             "Test - AUC:{:.4f}".format(auc(fpr_test, tpr_test))]
    plt.plot(fpr_train, tpr_train)
    plt.plot(fpr_test, tpr_test)
    plt.plot([0, 1], [0, 1], 'd--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(label, loc=4)
    plt.title("ROC curve")
    plt.show()


model_metrics(log_reg, train, test, y_train, y_test)

model_metrics(dtree, train, test, y_train, y_test)

model_metrics(LinearSVC, train, test, y_train, y_test)

model_metrics(xgbClassifier, train, test, y_train, y_test)

model_metrics(lgbmClassifier, train, test, y_train, y_test)