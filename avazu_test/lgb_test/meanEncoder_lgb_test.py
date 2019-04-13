# -*- coding: UTF-8 -*-
import logging
import time

import lightgbm as lgb
import pandas as pd
from sklearn import metrics

if __name__ == "__main__":
    start = time.time()
    df = pd.read_csv('../avazu_meanEncoder.csv', sep='\t')

    # 以6:2:2的比例分为训练集、验证集和测试集
    train = df[:int(df.shape[0] * 0.6)]
    val = df[int(df.shape[0] * 0.6):int(df.shape[0] * 0.8)]
    test = df[int(df.shape[0] * 0.8):]

    features = list(train.columns)
    features.remove("Label")

    X_train = train[features].values
    y_train = train['Label'].values
    X_val = val[features].values
    y_val = val['Label'].values
    X_test = test[features].values
    y_test = test['Label'].values

    param = {'boosting_type': 'gbdt',
             'objective': 'binary',
             'metric': 'binary_logloss',
             'learning_rate': 0.5,
             'max_depth': -1,
             'num_leaves': 80,
             'max_bin': 25,
             'min_data_in_leaf': 15,
             'feature_fraction': 1.0,
             'bagging_fraction': 1.0,
             'bagging_freq': 45,
             'lambda_l1': 0.4,
             'lambda_l2': 0.4,
             'min_split_gain': 0.4}

    num_round = 9999999
    trn_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val)
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                    early_stopping_rounds=10)

    # 训练集效果
    trainPred = clf.predict(X_train)
    logging.warning("训练集logloss: %.8f" % metrics.log_loss(y_train, trainPred))
    logging.warning("训练集auc: %.8f" % metrics.roc_auc_score(y_train, trainPred))
    # 验证集效果
    validPred = clf.predict(X_val)
    logging.warning("验证集logloss: %.8f" % metrics.log_loss(y_val, validPred))
    logging.warning("验证集auc: %.8f" % metrics.roc_auc_score(y_val, validPred))
    # 测试集效果
    testPred = clf.predict(X_test)
    logging.warning("测试集logloss: %.8f" % metrics.log_loss(y_test, testPred))
    logging.warning("测试集auc: %.8f" % metrics.roc_auc_score(y_test, testPred))
    logging.warning("early stop: %d" % clf.best_iteration)
