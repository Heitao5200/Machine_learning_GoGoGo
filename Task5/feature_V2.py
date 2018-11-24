import numpy as np
import pickle
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, \
    SoftImpute  # https://stackoverflow.com/questions/51695071/pip-install-ecos-error-microsoft-visual-c-14-0-is-required
from sklearn import preprocessing


path = "E:/MyPython/Machine_learning_GoGoGo/"
"""=====================================================================================================================
1 读取数据
"""
data = pd.read_csv(path + 'data_set/data.csv', encoding='gbk')
"""=====================================================================================================================
2 数据处理
"""
"""
2.1 对每一列的数据进行统计，如果这一列的数据每一个都不同，即判断为无关特征并且删去。
"""
for i in data.columns:
    count = data[i].count()
    if len(list(data[i].unique())) in [1, count, count - 1]:
        data.drop(i, axis=1, inplace=True)

"""
2.2 将每一个样本的缺失值的个数作为一个特征
"""
temp1 = data.isnull()
num = (temp1 == True).astype(bool).sum(axis=1)
is_null = DataFrame(list(zip(num)))
is_null = is_null.rename(columns={0: "is_null_num"})
data = pd.merge(data, is_null, left_index=True, right_index=True, how='outer')

"""
2.3 对reg_preference_for_trad 的处理  【映射】
    nan=0 境外=1 一线=5 二线=2 三线 =3 其他=4
"""
n = set(data['reg_preference_for_trad'])
dic = {}
for i, j in enumerate(n):
    dic[j] = i
data['reg_preference_for_trad'] = data['reg_preference_for_trad'].map(dic)

"""
2.4 对 id_name的处理  【映射】
"""
n = set(data['id_name'])
dic = {}
for i, j in enumerate(n):
    dic[j] = i
data['id_name'] = data['id_name'].map(dic)

"""
2.5 对 time 的处理  【删除】
"""

X_date = pd.DataFrame()
X_date['latest_query_time_month'] = pd.to_datetime(data['latest_query_time']).dt.month  # 月份
X_date['latest_query_time_weekday'] = pd.to_datetime(data['latest_query_time']).dt.weekday  # 星期几

X_date['loans_latest_time_month'] = pd.to_datetime(data['loans_latest_time']).dt.month  # 月份
X_date['loans_latest_time_weekday'] = pd.to_datetime(data['loans_latest_time']).dt.weekday  # 星期几

# X_date.fillna(X_date.median(), inplace=True)
data.drop(["latest_query_time"], axis=1, inplace=True)
data.drop(["loans_latest_time"], axis=1, inplace=True)
data = pd.concat([data, X_date], axis=1, sort=False)

# """
# 2.6 使用众数填充
# """
# filter_feature = ['status'] # 过滤无用的维度
# features = []
# for x in data.columns: # 取特征
#     if x not in filter_feature:
#         features.append(x)
# data.fillna(data.mode(),inplace=True) # 填充众数,该数据缺失太多众数出现为nan的情况
# features_mode = {}
# for f in features:
#     features_mode[f] = list(data[f].dropna().mode().values)[0]
# data.fillna(features_mode,inplace=True)



"""
2.5 使用KNN对数据填充
"""
filter_feature = ['status']  # 取预测值
features = []
for x in data.columns:  # 取特征
    if x not in filter_feature:
        features.append(x)
data_x = data[features]
data_x = pd.DataFrame(KNN(k=3).fit_transform(data_x), columns=features)
data_y = data['status']


"""
2.6 数据归一化
"""
min_max_scale = preprocessing.MinMaxScaler()
data_x = min_max_scale.fit_transform(data_x)


"""=====================================================================================================================
3 数据切分
"""
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, random_state=1)  # 划分训练集、测试集

"""=====================================================================================================================
4 特征保存
"""
print("3 保存至本地")
data = (X_train, X_test, y_train, y_test)
fp = open(path + 'feature/feature_V3.pkl', 'wb')
pickle.dump(data, fp)
fp.close()

linreg = LogisticRegression()
linreg.fit(X_train, y_train)  # 模型训练

# y_pred = linreg.predict(X_train)  # 模型预测
# print("训练集", countF1(y_train.values, y_pred))
#
# y_pred = linreg.predict(X_test)  # 模型预测
# print("测试集", countF1(y_test.values, y_pred))

print("Train_Score ：{}".format(linreg.score(X_train, y_train)))
print("Test_Score ：{}".format(linreg.score(X_test, y_test)))