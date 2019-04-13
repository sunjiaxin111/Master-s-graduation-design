# -*- coding: UTF-8 -*-
import logging
import time

import lightgbm as lgb
import pandas as pd
from sklearn import metrics

if __name__ == "__main__":
    start = time.time()
    df = pd.read_csv('criteo_meanEncoder.csv', sep='\t')

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
             'num_leaves': 120,
             'max_bin': 5,
             'min_data_in_leaf': 170,
             'feature_fraction': 1.0,
             'bagging_fraction': 0.4,
             'bagging_freq': 0,
             'lambda_l1': 0.9,
             'lambda_l2': 0.1,
             'min_split_gain': 1.0}

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

    df_result_train = pd.DataFrame(train['Label'], columns=['Label'])
    df_result_val = pd.DataFrame(val['Label'], columns=['Label'])
    df_result_test = pd.DataFrame(test['Label'], columns=['Label'])
    y_pred_train = clf.predict(train[features], pred_leaf=True).reshape(train.shape[0], -1)
    y_pred_val = clf.predict(val[features], pred_leaf=True).reshape(val.shape[0], -1)
    y_pred_test = clf.predict(test[features], pred_leaf=True).reshape(test.shape[0], -1)
    for i in range(1, y_pred_train.shape[1] + 1):
        # 提取叶子节点特征
        df_result_train['leaf' + str(i)] = y_pred_train[:, i - 1]
        df_result_val['leaf' + str(i)] = y_pred_val[:, i - 1]
        df_result_test['leaf' + str(i)] = y_pred_test[:, i - 1]

        ##############################################################

        # 提取交叉熵特征
        # 这里计算一下交叉熵，为了避免过拟合，只使用训练集信息来计算
        # 用dict记录一下
        dic = {}
        df_result_train['prob'] = clf.predict(train[features], num_iteration=i)
        # 我只需要取值空间
        leave_ids = df_result_train['leaf' + str(i)].unique()
        for leave_id in leave_ids:
            # 计算对应的交叉熵
            temp_cross_entropy = metrics.log_loss(
                df_result_train[df_result_train['leaf' + str(i)] == leave_id]['Label'],
                df_result_train[df_result_train['leaf' + str(i)] == leave_id]['prob'], labels=[1, 0])
            dic[leave_id] = temp_cross_entropy
        # 删除临时的prob列
        df_result_train = df_result_train.drop(['prob'], axis=1)
        # 根据dic在数据集中新增交叉熵列
        df_result_train['cross_entropy' + str(i)] = df_result_train['leaf' + str(i)].map(lambda x: dic[x])
        df_result_val['cross_entropy' + str(i)] = df_result_val['leaf' + str(i)].map(lambda x: dic[x])
        df_result_test['cross_entropy' + str(i)] = df_result_test['leaf' + str(i)].map(lambda x: dic[x])

        #############################################################

        # 提取正例率特征
        # 使用df_result_train计算正例率
        dic = {}
        leave_ids = df_result_train['leaf' + str(i)].unique()
        for leave_id in leave_ids:
            # 计算对应的正例率
            temp_pos_ratio = df_result_train[df_result_train['leaf' + str(i)] == leave_id]['Label'].mean()
            dic[leave_id] = temp_pos_ratio

        # 根据dic在数据集中新增交叉熵列
        df_result_train['pos_ratio' + str(i)] = df_result_train['leaf' + str(i)].map(lambda x: dic[x])
        df_result_val['pos_ratio' + str(i)] = df_result_val['leaf' + str(i)].map(lambda x: dic[x])
        df_result_test['pos_ratio' + str(i)] = df_result_test['leaf' + str(i)].map(lambda x: dic[x])

    # 把训练集和验证集合并为一个
    df = pd.concat([df_result_train, df_result_val, df_result_test])

    df.to_csv("criteo_lgbEncoder.csv", index=None, sep='\t')
