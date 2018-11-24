#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : Lightgbm.py
# @Date    : 2018-11-17
# @Author  : 黑桃
# @Software: PyCharm 
import lightgbm as lgb
import pickle
from sklearn import metrics
from sklearn.externals import joblib

path = "E:/MyPython/Machine_learning_GoGoGo/"
"""=====================================================================================================================
1 读取特征
"""
print("0 读取特征")
f = open(path + 'feature/feature_V3.pkl', 'rb')
train, test, y_train,y_test= pickle.load(f)
f.close()

"""【将数据格式转换成lgb模型所需的格式】"""
lgb_train = lgb.Dataset(train, y_train)
lgb_eval = lgb.Dataset(test, y_test, reference=lgb_train)
"""=====================================================================================================================
2 设置模型训练参数
"""
"""【LGB_自带接口的参数】"""
params = {
'task': 'train',
'boosting_type': 'gbdt',
'objective': 'binary',
'metric': {'l2', 'auc'},
'num_leaves': 31,
'learning_rate': 0.05,
'feature_fraction': 0.9,
'bagging_fraction': 0.8,
'bagging_freq': 5,
'verbose': 0
}

"""=====================================================================================================================
3 模型训练
"""
##分类使用的是 LGBClassifier
##回归使用的是 LGBRegression
"""【LGB_自带接口训练】"""
model_lgb = lgb.train(params,lgb_train,num_boost_round=100,valid_sets=lgb_eval,early_stopping_rounds=10)

# y_lgb_proba = model_lgb.predict_proba(test)
"""【LGB_sklearn接口训练】"""
lgb_sklearn = lgb.LGBMClassifier(learning_rate=0.1,
    max_bin=150,
    num_leaves=32,
    max_depth=11,
    reg_alpha=0.1,
    reg_lambda=0.2,
    # objective='multiclass',
    n_estimators=300,)
lgb_sklearn.fit(train,y_train)

"""【保存模型】"""
print('3 保存模型')
joblib.dump(model_lgb, path + "model/model_file/lgb.pkl")
joblib.dump(lgb_sklearn, path + "model/model_file/lgb_sklearn.pkl")
"""=====================================================================================================================
4 模型预测
"""
"""【LGB_自带接口预测】"""
y_lgb_pre = model_lgb.predict(test, num_iteration=model_lgb.best_iteration)

"""【LGB_sklearn接口预测】"""
y_sklearn_pre= lgb_sklearn.predict(test)
y_sklearn_proba= lgb_sklearn.predict_proba(test)[:,1]
"""=====================================================================================================================
5 模型评分
"""
print('LGB_自带接口(predict)    AUC Score:', metrics.roc_auc_score(y_test, y_lgb_pre) )
print('LGB_sklearn接口(proba)    AUC Score:', metrics.roc_auc_score(y_test, y_sklearn_proba) )
print('LGB_sklearn接口(predict)    AUC Score:', metrics.roc_auc_score(y_test, y_sklearn_pre) )



## [lightgbm的原生版本与sklearn 接口版本对比](https://blog.csdn.net/PIPIXIU/article/details/82709899)
## [lightGBM原理、改进简述](https://blog.csdn.net/niaolianjiulin/article/details/76584785)
## [LightGBM 如何调参](https://blog.csdn.net/aliceyangxi1987/article/details/80711014)




