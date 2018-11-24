#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : SVM.py
# @Date    : 2018-11-14
# @Author  : 黑桃
# @Software: PyCharm


import pickle
import pandas as pd #数据分析
from pandas import Series,DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
print("开始......")
t_start = time.time()
path = "E:/MyPython/Machine_learning_GoGoGo/"
"""=====================================================================================================================
1 读取数据
"""
print("数据预处理")
data = pd.read_csv(path + 'data_set/data.csv',encoding='gbk')

"""=====================================================================================================================
2 数据处理
"""
"""将每一个样本的缺失值的个数作为一个特征"""
temp1=data.isnull()
num=(temp1 == True).astype(bool).sum(axis=1)
is_null=DataFrame(list(zip(num)))
is_null=is_null.rename(columns={0:"is_null_num"})
data = pd.merge(data,is_null,left_index = True, right_index = True, how = 'outer')

"""
1.1 缺失值用100填充
"""
data=DataFrame(data.fillna(100))


"""
1.2 对reg_preference_for_trad 的处理  【映射】
    nan=0 境外=1 一线=5 二线=2 三线 =3 其他=4
"""
n=set(data['reg_preference_for_trad'])
dic={}
for i,j in enumerate(n):
    dic[j]=i
data['reg_preference_for_trad'] = data['reg_preference_for_trad'].map(dic)


"""
1.2 对source 的处理  【映射】
"""
n=set(data['source'])
dic={}
for i,j in enumerate(n):
    dic[j]=i
data['source'] = data['source'].map(dic)


"""
1.3 对bank_card_no 的处理  【映射】
"""
n=set(data['bank_card_no'])
dic={}
for i,j in enumerate(n):
    dic[j]=i
data['bank_card_no'] = data['bank_card_no'].map(dic)

"""
1.2 对 id_name的处理  【映射】
"""
n=set(data['id_name'])
dic={}
for i,j in enumerate(n):
    dic[j]=i
data['id_name'] = data['id_name'].map(dic)

"""
1.2 对 time 的处理  【删除】
"""
data.drop(["latest_query_time"],axis=1,inplace=True)
data.drop(["loans_latest_time"],axis=1,inplace=True)


## ['trade_no'] 这一列格式不对 可以选择直接删除 或者转换  这里【直接删除】
data = data.drop(['trade_no'],axis=1)

status = data.status
# """=====================================================================================================================
# 4 time时间归一化 小时
# """
# data['time'] = pd.to_datetime(data['time'])
# time_now = data['time'].apply(lambda x:int((x-datetime(2018,11,14,0,0,0)).seconds/3600))
# data['time']= time_now

"""=====================================================================================================================
2 划分训练集和验证集，验证集比例为test_size
"""
print("划分训练集和验证集，验证集比例为test_size")
train, test = train_test_split(data, test_size=0.3, random_state=666)


"""=====================================================================================================================
3 分标签和 训练数据
"""
y_train= train.status
train.drop(["status"],axis=1,inplace=True)

y_test= test.status
test.drop(["status"],axis=1,inplace=True)
"""
1.4标准化数据，方差为1，均值为零
"""
standardScaler = StandardScaler()
test = standardScaler.fit_transform(test)
train = standardScaler.fit_transform(train)





print("3 保存至本地")
data = (train, test, y_train,y_test)
fp = open(path + 'feature/feature_V1.pkl', 'wb')
pickle.dump(data, fp)
fp.close()