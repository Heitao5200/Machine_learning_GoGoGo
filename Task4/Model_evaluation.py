#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : Model_evaluation.py
# @Date    : 2018-11-20
# @Author  : 黑桃
# @Software: PyCharm 

import pickle
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve

path = "E:/MyPython/Machine_learning_GoGoGo/"
"""=====================================================================================================================
1 读取特征
"""
print("0 读取特征")
f = open(path + 'feature/feature_V3.pkl', 'rb')
train, test, y_train,y_test= pickle.load(f)
f.close()

"""=====================================================================================================================
2 读取模型
"""
print("1 读取模型")
SVM_linear = joblib.load( path + "model/model_file/SVM_linear.pkl")
SVM_poly = joblib.load( path + "model/model_file/SVM_poly.pkl")
SVM_rbf = joblib.load( path + "model/model_file/SVM_rbf.pkl")
SVM_sigmoid = joblib.load( path + "model/model_file/SVM_sigmoid.pkl")
lg_120 = joblib.load( path + "model/model_file/lg_120.pkl")
DT = joblib.load( path + "model/model_file/DT.pkl")
xgb_sklearn = joblib.load( path + "model/model_file/xgb_sklearn.pkl")
lgb_sklearn = joblib.load( path + "model/model_file/lgb_sklearn.pkl")
xgb = joblib.load( path + "model/model_file/xgb.pkl")
lgb = joblib.load( path + "model/model_file/lgb.pkl")




"""=====================================================================================================================
3 模型评估
"""

def model_evalua(clf, X_train, X_test, y_train, y_test,name):
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    y_train_pred_proba = clf.predict_proba(X_train)[:, 1]
    y_test_pred_proba = clf.predict_proba(X_test)[:, 1]
    """【Score】"""
    print(name + '_Score')
    print(name + "_Train_Score ：{:.4f}".format(clf.score(X_train, y_train)))
    print(name + "_Test_Score ：{:.4f}".format(clf.score(X_test, y_test)))

    """【AUC Score】"""
    print(name+'_AUC Score')
    print(name+"_Train_AUC Score ：{:.4f}".format(roc_auc_score(y_train, y_train_pred)))
    print(name+"_Test_AUC Score ：{:.4f}".format(roc_auc_score(y_test, y_test_pred)))

    """【准确性】"""
    print(name+'_准确性：')
    print(name+'_Train_准确性：{:.4f}'.format(accuracy_score(y_train, y_train_pred)))
    print(name+'_Test_准确性：{:.4f}'.format(accuracy_score(y_test, y_test_pred)))

    """【召回率】"""
    print(name+'_召回率：')
    print(name+'_Train_召回率：{:.4f}'.format(recall_score(y_train, y_train_pred)))
    print(name+'_Test_召回率：{:.4f}'.format(recall_score(y_test, y_test_pred)))

    """【f1_score】"""
    print(name+'_f1_score：')
    print(name+'_Train_f1_score：{:.4f}'.format(f1_score(y_train, y_train_pred)))
    print(name+'_Test_f1_score：{:.4f}'.format(f1_score(y_test, y_test_pred)))

    #描绘 ROC 曲线
    fpr_tr, tpr_tr, _ = roc_curve(y_train, y_train_pred_proba)
    fpr_te, tpr_te, _ = roc_curve(y_test, y_test_pred_proba)
    # KS
    print(name+'_KS：')
    print(name+'_Train：{:.4f}'.format(max(abs((fpr_tr - tpr_tr)))))
    print(name+'_Test：{:.4f}'.format(max(abs((fpr_te - tpr_te)))))
    plt.plot(fpr_tr, tpr_tr, 'r-',
             label = name+"_Train:AUC: {:.3f} KS:{:.3f}".format(roc_auc_score(y_train, y_train_pred_proba),
                                                max(abs((fpr_tr - tpr_tr)))))
    plt.plot(fpr_te, tpr_te, 'g-',
             label=name+"_Test:AUC: {:.3f} KS:{:.3f}".format(roc_auc_score(y_test, y_test_pred_proba),
                                                       max(abs((fpr_tr - tpr_tr)))))
    plt.plot([0, 1], [0, 1], 'd--')
    plt.legend(loc='best')
    plt.title(name+"_ROC curse")
    plt.savefig(path +'picture/'+name+'.jpg')
    plt.show()
print('-------------------SVM_linear-------------------')
model_evalua(SVM_linear, train, test, y_train, y_test,'SVM_linear')
#
print('-------------------SVM_poly-------------------：')
model_evalua(SVM_poly, train, test, y_train, y_test,'SVM_poly')

print('-------------------SVM_rbf-------------------：')
model_evalua(SVM_rbf, train, test, y_train, y_test,'SVM_rbf')

print('-------------------SVM_sigmoid-------------------：')
model_evalua(SVM_sigmoid, train, test, y_train, y_test,'SVM_sigmoid')

print('-------------------lg_120-------------------')
model_evalua(lg_120, train, test, y_train, y_test,'lg_120')

print('-------------------DT-------------------')
model_evalua(DT, train, test, y_train, y_test,'DT')

print('-------------------xgb_sklearn-------------------')
model_evalua(xgb_sklearn, train, test, y_train, y_test,'xgb_sklearn')

print('-------------------xgb-------------------')
# model_evalua(xgb, train, test, y_train, y_test,'xgb')

print('-------------------lgb_sklearn-------------------')
model_evalua(lgb_sklearn, train, test, y_train, y_test,'lgb_sklearn')
print('-------------------lgb-------------------')
# model_evalua(lgb, train, test, y_train, y_test,'lgb')



