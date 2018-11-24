#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : SVM.py
# @Date    : 2018-11-14
# @Author  : 黑桃
# @Software: PyCharm 


import pickle
import time
from  sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.externals import joblib
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
2 模型训练
"""
print("支持向量机模型训练")



SVM_linear = svm.SVC(kernel = 'linear', probability=True).fit(train, y_train)
SVM_poly = svm.SVC(kernel = 'poly', probability=True).fit(train, y_train)
SVM_rbf = svm.SVC(probability=True).fit(train, y_train)
SVM_sigmoid = svm.SVC(kernel = 'sigmoid',probability=True).fit(train, y_train)




"""【保存模型】"""
print('3 保存模型')
joblib.dump(SVM_linear, path + "model/model_file/SVM_linear.pkl")
joblib.dump(SVM_poly, path + "model/model_file/SVM_poly.pkl")
joblib.dump(SVM_rbf, path + "model/model_file/SVM_rbf.pkl")
joblib.dump(SVM_sigmoid, path + "model/model_file/SVM_sigmoid.pkl")

"""=====================================================================================================================
3 模型预测
"""
SVM_linear_test_pre = SVM_linear.predict(test)
SVM_poly_test_pre = SVM_poly.predict(test)
SVM_rbf_test_pre = SVM_rbf.predict(test)
SVM_sigmoid_test_pre = SVM_sigmoid.predict(test)


"""=====================================================================================================================
4 模型评分
"""
print("-----------------f1分数-----------------")
print("SVM_linear_test_pre  f1分数：{}".format(f1_score(y_test, SVM_linear_test_pre, average='macro')))
print("SVM_poly_test_pre    f1分数：{}".format(f1_score(y_test, SVM_poly_test_pre, average='macro')))
print("SVM_rbf_test_pre f1分数：{}".format(f1_score(y_test, SVM_rbf_test_pre, average='macro')))
print("SVM_sigmoid_test_pre f1分数：{}".format(f1_score(y_test, SVM_sigmoid_test_pre, average='macro')))
# r2 = r2_score(y_test, y_test_pre)
print("-----------------验证集分数-----------------")

print("SVM_linear   验证集分数：{}".format(SVM_linear.score(test, y_test)))
print("SVM_poly   验证集分数：{}".format(SVM_poly.score(test, y_test)))
print("SVM_rbf   验证集分数：{}".format(SVM_rbf.score(test, y_test)))
print("SVM_sigmoid   验证集分数：{}".format(SVM_sigmoid.score(test, y_test)))
end_time = time.time()
print("程序结束！！ 耗时%s"%(end_time-t_start))