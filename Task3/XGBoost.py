#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : XGBoost.py
# @Date    : 2018-11-17
# @Author  : 黑桃
# @Software: PyCharm
import xgboost as xgb
import time
from sklearn.externals import joblib
from xgboost.sklearn import XGBClassifier
import pickle
from sklearn import metrics
start_time = time.time()

path = "E:/MyPython/Machine_learning_GoGoGo/"
"""=====================================================================================================================
1 读取特征
"""
print("0 读取特征")
f = open(path + 'feature/feature_V3.pkl', 'rb')
train, test, y_train,y_test= pickle.load(f)
f.close()
"""【将数据格式转换成xgb模型所需的格式】"""
xgb_val = xgb.DMatrix(test,label=y_test)
xgb_train = xgb.DMatrix(train, label=y_train)
xgb_test = xgb.DMatrix(test)
"""=====================================================================================================================
2 设置模型训练参数
"""
## XGB自带接口
params={
'booster':'gbtree',
'objective': 'reg:linear', #多分类的问题
'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':12, # 构建树的深度，越大越容易过拟合
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':3,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.007, # 如同学习率
'seed':1000,
'nthread':7,# cpu 线程数
#'eval_metric': 'auc'
}
plst = list(params.items())## 转化为list   为什么要转化？
num_rounds =5000 # 设置迭代次数


#sklearn接口

##分类使用的是 XGBClassifier
##回归使用的是 XGBRegression
clf = XGBClassifier(
    n_estimators=30,#三十棵树
    learning_rate =0.3,
    max_depth=3,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=12,
    scale_pos_weight=1,
    reg_lambda=1,
    seed=27)

watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]


"""=====================================================================================================================
3 模型训练
"""

# training model
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
# 使用XGBoost有自带接口
"""【使用XGBoost自带接口训练】"""
model_xgb = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)

## Scikit-Learn接口
"""【Scikit-Learn接口训练】"""
model_xgb_sklearn=clf.fit(train, y_train)

"""【保存模型】"""
print('3 保存模型')
joblib.dump(model_xgb, path + "model/model_file/xgb.pkl")
joblib.dump(model_xgb_sklearn, path + "model/model_file/xgb_sklearn.pkl")

"""=====================================================================================================================
4 模型预测
"""
"""【使用XGBoost自带接口预测】"""
y_xgb = model_xgb.predict(xgb_test)
"""【Scikit-Learn接口预测】"""
y_sklearn_pre= model_xgb_sklearn.predict(test)
y_sklearn_proba= model_xgb_sklearn.predict_proba(test)[:,1]

"""=====================================================================================================================
5 模型评分
"""
print("XGBoost_自带接口(predict) : %s" % y_xgb)
print("XGBoost_sklearn接口(proba): %s" % y_sklearn_proba)
print("XGBoost_sklearn接口(predict)  : %s" % y_sklearn_pre)

# print("XGBoost_自带接口(predict)     AUC Score : %f" % metrics.roc_auc_score(y_test, y_xgb))
# print("XGBoost_sklearn接口(proba)  AUC Score : %f" % metrics.roc_auc_score(y_test, y_sklearn_proba))
# print("XGBoost_sklearn接口(predict) AUC Score : %f" % metrics.roc_auc_score(y_test, y_sklearn_pre))
"""【roc_auc_score】"""
#直接根据真实值（必须是二值）、预测值（可以是0/1,也可以是proba值）计算出auc值，中间过程的roc计算省略。
# f1 = f1_score(y_test, predictions, average='macro')
print("XGBoost_自带接口(predict)           AUC Score ：{}".format(metrics.roc_auc_score(y_test, y_xgb)))
print("XGBoost_sklearn接口(proba)         AUC Score : {}".format(metrics.roc_auc_score(y_test, y_sklearn_proba)))
print("XGBoost_sklearn接口(predict)       AUC Score ：{}".format(metrics.roc_auc_score(y_test, y_sklearn_pre)))

## [机器学习xgboost实战—手写数字识别 （DMatrix）](https://blog.csdn.net/u010159842/article/details/78053669)
## [Windows下在Anaconda3中安装python版的XGBoost库](https://blog.csdn.net/zz860890410/article/details/78682041)
## [XGBoost Plotting API以及GBDT组合特征实践](https://blog.csdn.net/sb19931201/article/details/65445514)