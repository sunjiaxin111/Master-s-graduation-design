# -*- coding: UTF-8 -*-
import time

import pandas as pd
import xlearn as xl
from sklearn import preprocessing, metrics

if __name__ == "__main__":
    start = time.time()
    df = pd.read_csv('../avazu_labelEncoder.csv', sep='\t')
    # 类别类型
    cate_features = list(df.columns)
    cate_features.remove("Label")

    # 连续型特征
    continuous_cols = []

    # one-hot编码
    df = pd.get_dummies(df, columns=cate_features)

    # 以6:2:2的比例分为训练集、验证集和测试集
    train = df[:int(df.shape[0] * 0.6)]
    val = df[int(df.shape[0] * 0.6):int(df.shape[0] * 0.8)]
    test = df[int(df.shape[0] * 0.8):]

    y_train = train['Label']
    y_valid = val['Label']
    y_test = test['Label']

    # 标准化
    for feat in continuous_cols:
        scaler = preprocessing.StandardScaler().fit(train[feat].values.reshape(-1, 1))
        train[feat] = scaler.transform(train[feat].values.reshape(-1, 1))
        val[feat] = scaler.transform(val[feat].values.reshape(-1, 1))
        test[feat] = scaler.transform(test[feat].values.reshape(-1, 1))

    new_path = "FM/"

    # 把训练集、验证集、测试集存储起来
    train.to_csv(new_path + 'train_fm.csv', index=None, sep='\t', header=None)
    val.to_csv(new_path + 'valid_fm.csv', index=None, sep='\t', header=None)
    test.to_csv(new_path + 'test_fm.csv', index=None, sep='\t', header=None)

    fm_model = xl.create_fm()
    fm_model.setTrain(new_path + 'train_fm.csv')
    fm_model.setValidate(new_path + 'valid_fm.csv')

    param = {'task': 'binary',
             'lr': 0.2,
             'lambda': 0.002,
             'metric': 'auc',
             'k': 8,
             'epoch': 999999
             }
    fm_model.disableLockFree()
    fm_model.fit(param, new_path + "fm_model.out")

    # 预测
    fm_model.setSigmoid()
    fm_model.setTest(new_path + 'train_fm.csv')
    fm_model.predict(new_path + 'fm_model.out', new_path + "train_pred.txt")
    fm_model.setTest(new_path + 'valid_fm.csv')
    fm_model.predict(new_path + 'fm_model.out', new_path + "valid_pred.txt")
    fm_model.setTest(new_path + 'test_fm.csv')
    fm_model.predict(new_path + 'fm_model.out', new_path + "test_pred.txt")

    # 训练集效果
    trainPred = pd.read_csv(new_path + "train_pred.txt", header=None)
    print("训练集logloss:", metrics.log_loss(y_train, trainPred))
    print("训练集auc:", metrics.roc_auc_score(y_train, trainPred))
    # 验证集效果
    validPred = pd.read_csv(new_path + "valid_pred.txt", header=None)
    print("验证集logloss:", metrics.log_loss(y_valid, validPred))
    print("验证集auc:", metrics.roc_auc_score(y_valid, validPred))
    # 测试集效果
    testPred = pd.read_csv(new_path + "test_pred.txt", header=None)
    print("测试集logloss:", metrics.log_loss(y_test, testPred))
    print("测试集auc:", metrics.roc_auc_score(y_test, testPred))
