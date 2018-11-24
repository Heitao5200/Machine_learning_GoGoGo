#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import *   #Additional     scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search
import pickle
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
path = "E:/MyPython/Machine_learning_GoGoGo/"
"""=====================================================================================================================
1 读取数据
"""
print("0 读取特征")
f = open(path + 'feature/feature_V3.pkl', 'rb')
train,test,train_y,test_y = pickle.load(f)
f.close()


def XGBmodelfit(alg, X_train, Y_train,X_test=None,Y_test=None,X_predictions=None,useTrainCV=True, cv_folds=5, early_stopping_rounds=200):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=Y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_stdv=False)
    alg.set_params(n_estimators=cvresult.shape[0])

    # 训练模型
    alg.fit(X_train, Y_train, eval_metric='auc')

    # 预测结果:
    dtrain_predictions = alg.predict(X_test)  # 输出 0 或 1
    # dtrain_predprob = alg.predict_proba(X_test)[:,1]   #输出概率

    # 打印报告信息:
    print("\nModel Report")
    print("Accuracy  (Train) : %.4g" % metrics.accuracy_score(Y_test, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(Y_test, dtrain_predictions))
    print(alg)
    print("the best:")
    print(cvresult.shape[0])
    plot_importance(alg)
    plt.show()

feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

dataset_X, dataset_Y = return_data.return_tarin_data()

X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y,
                                                    test_size=0.2,
                                                    random_state=45)

xgb1 = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)

# XGBmodelfit(xgb1,X_train,y_train,X_test,y_test)

param_grid = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}
# param_grid = {
#  'max_depth':[7,8],
#  'min_child_weight':[4,5]
# }
gsearch1 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1, n_estimators=140, max_depth=9,
    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
    objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
    param_grid=param_grid, cv=5)
gsearch1.fit(X_train, y_train)
print(gsearch1.best_params_, gsearch1.best_score_)
















