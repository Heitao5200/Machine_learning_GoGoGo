#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : GrideSearchCV.py
# @Date    : 2018-11-23
# @Author  : 黑桃
# @Software: PyCharm
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn import svm
import lightgbm as lgb
from sklearn.model_selection import *
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
path = "E:/MyPython/Machine_learning_GoGoGo/"
"""=====================================================================================================================
1 读取数据
"""
print("0 读取特征")
f = open(path + 'feature/feature_V3.pkl', 'rb')
X_train,X_test,y_train,y_test = pickle.load(f)
f.close()
"""
3.1 交叉验证方式
"""
## 对交叉验证方式进行指定，如验证次数，训练集测试集划分比例等
kf = KFold(n_splits=5, random_state=1)
loo = LeaveOneOut()#将数据集分成训练集和测试集，测试集包含一个样本，训练集包含n-1个样本
lpo = LeavePOut(p=2000)## #将数据集分成训练集和测试集，测试集包含p个样本，训练集包含n-p个样本
ss= ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
tss = TimeSeriesSplit(n_splits=5)
## 下面的几种分组的交叉验证方式还没弄懂
logo = LeaveOneGroupOut()
lpgo = LeavePGroupsOut(n_groups=3)
gss = GroupShuffleSplit(n_splits=5, test_size=.2, random_state=0)
gkf = GroupKFold(n_splits=2)



"""=====================================================================================================================
2 模型参数设置
"""
"""【SVM】"""
SVM_linear = svm.SVC(kernel = 'linear', probability=True)
SVM_poly = svm.SVC(kernel = 'poly', probability=True)
SVM_rbf = svm.SVC(kernel = 'rbf',probability=True)
SVM_sigmoid = svm.SVC(kernel = 'sigmoid',probability=True)

SVM_param_grid = {"C":[0.001,0.01,0.1,1,10,100]}


"""【LG】"""
LG = LogisticRegression()
LG_param_grid = {"C":[0.001,0.01,0.1,1,10,100]}


"""【DT】"""
DT = DecisionTreeClassifier()
DT_param_grid = {'max_depth':range(1,5)}
params = {'max_depth':range(1,21),'criterion':np.array(['entropy','gini'])}


"""【XGB_sklearn】"""
# XGB_sklearn = XGBClassifier(n_estimators=30,#三十棵树
#     learning_rate =0.3,
#     max_depth=3,
#     min_child_weight=1,
#     gamma=0.3,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective= 'binary:logistic',
#     nthread=12,
#     scale_pos_weight=1,
#     reg_lambda=1,
#     seed=27)
XGB_sklearn = XGBClassifier()
XGB_sklearn_param_grid = {"max_depth":[1,10,100]}
XGB_sklearn_param_grid1 = {"max_depth":[1,10,100],
    'min_child_weight': [6, 8, 10, 12],
    'gamma':[i/10.0 for i in range(0,5)],
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)],
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
                          }

# param_test4 = {
#     'min_child_weight':[6,8,10,12],
#     'gamma':[i/10.0 for i in range(0,5)],
#     'subsample':[i/10.0 for i in range(6,10)],
#     'colsample_bytree':[i/10.0 for i in range(6,10)],
#     'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
# }



"""【LGB_sklearn】"""
LGB_sklearn = lgb.LGBMClassifier()
# LGB_sklearn = lgb.LGBMClassifier(learning_rate=0.1,
#     max_bin=150,
#     num_leaves=32,
#     max_depth=11,
#     reg_alpha=0.1,
#     reg_lambda=0.2,
#     # objective='multiclass',
#     n_estimators=300,)
LGB_sklearn_param_grid = {"max_depth":[1,10,100]}

def GridSearch_CV(clf,param_grid,cv,name):
    grid_search = GridSearchCV(clf,param_grid,cv=cv) #实例化一个GridSearchCV类
    grid_search.fit(X_train,y_train) #训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
    print(name+"_Test set score:{}".format(grid_search.score(X_test,y_test)))
    print(name+"_Best parameters:{}".format(grid_search.best_params_))
    print(name+"_Best score on train set:{}".format(grid_search.best_score_))

GridSearch_CV(LGB_sklearn,LGB_sklearn_param_grid,ss,"LGB_sklearn")
GridSearch_CV(LG,LG_param_grid,ss,"LG")
GridSearch_CV(SVM_linear,SVM_param_grid,ss,"SVM_linear")
GridSearch_CV(SVM_rbf,SVM_param_grid,ss,"SVM_rbf")
GridSearch_CV(SVM_sigmoid,SVM_param_grid,ss,"SVM_sigmoid")
GridSearch_CV(SVM_poly,SVM_param_grid,ss,"SVM_poly")
GridSearch_CV(DT,DT_param_grid,ss,"DT")
GridSearch_CV(XGB_sklearn,XGB_sklearn_param_grid,ss,"XGB_sklearn")







