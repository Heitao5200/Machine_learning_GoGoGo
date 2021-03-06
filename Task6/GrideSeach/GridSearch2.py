#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : GridSearch2.py
# @Date    : 2018-11-23
# @Author  : 黑桃
# @Software: PyCharm 
import pickle
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import time

print("开始......")
t_start = time.time()

path = "E:/MyPython/Machine_learning_GoGoGo/"
"""=====================================================================================================================
1 读取数据
"""
print("0 读取特征")
f = open(path + 'feature/feature_V3.pkl', 'rb')
train_X,test_X,train_y,test_y = pickle.load(f)
f.close()

featurer = TfidfVectorizer(ngram_range=(1,2),min_df=3 )
classifier = LinearSVC()
pipeline = Pipeline([('tfidf', featurer),('clf', classifier)])

parameters = {'tfidf__ngram_range': ((1, 2), (1, 3)),
              'tfidf__min_df': (4, 6),
              'tfidf__max_df':(0.7, 0.9),
              'clf__C': (1.0, 2.0)}

skf = StratifiedKFold(n_splits=5, random_state=1)
gs = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs= 1, scoring='f1_macro', cv=skf, verbose=3)
gs.fit(df_train_x, df_train_y)

""" 
打印最优的参数值
"""
print("Best score: %0.3f" % gs.best_score_)
best_parameters = gs.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("/t%s: %r" % (param_name, best_parameters[param_name]))

x_test = gs.best_estimator_.named_steps['tfidf'].transform(df_test['word_seg'])

"""
根据上面训练好的分类器对测试集的每个样本进行预测
"""
y_test = gs.best_estimator_.named_steps['clf'].predict(x_test)

""" 
将测试集的预测结果保存至本地
"""
df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id', 'class']]
df_result.to_csv('../../result/beginner.csv', index=False)
t_end = time.time()
print("训练结束，耗时:{}s".format(t_end-t_start))




