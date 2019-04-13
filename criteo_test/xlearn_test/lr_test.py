# -*- coding: UTF-8 -*-
import time

import pandas as pd
import xlearn as xl
from sklearn import preprocessing, metrics

if __name__ == "__main__":
    start = time.time()
    df = pd.read_csv('../criteo_labelEncoder.csv', sep='\t')
    # 类别类型
    cate_features = ['C1', 'C2', 'C3', 'C4', 'C5',
                     'C6', 'C7', 'C8', 'C9', 'C10',
                     'C11', 'C12', 'C13', 'C14', 'C15',
                     'C16', 'C17', 'C18', 'C19', 'C20',
                     'C21', 'C22', 'C23', 'C24', 'C25',
                     'C26']

    # 连续型特征
    continuous_cols = ['I1', 'I2', 'I3', 'I4', 'I5',
                       'I6', 'I7', 'I8', 'I9', 'I10',
                       'I11', 'I12', 'I13']

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

    new_path = "LR/"

    # 把训练集、验证集、测试集存储起来
    train.to_csv(new_path + 'train_lr.csv', index=None, sep='\t', header=None)
    val.to_csv(new_path + 'valid_lr.csv', index=None, sep='\t', header=None)
    test.to_csv(new_path + 'test_lr.csv', index=None, sep='\t', header=None)

    lr_model = xl.create_linear()
    lr_model.setTrain(new_path + 'train_lr.csv')
    lr_model.setValidate(new_path + 'valid_lr.csv')

    param = {'task': 'binary',
             'lr': 0.2,
             'lambda': 0.002,
             'metric': 'auc',
             'k': 8,
             'epoch': 999999
             }
    lr_model.disableLockFree()
    lr_model.fit(param, new_path + "lr_model.out")

    # 预测
    lr_model.setSigmoid()
    lr_model.setTest(new_path + 'train_lr.csv')
    lr_model.predict(new_path + 'lr_model.out', new_path + "train_pred.txt")
    lr_model.setTest(new_path + 'valid_lr.csv')
    lr_model.predict(new_path + 'lr_model.out', new_path + "valid_pred.txt")
    lr_model.setTest(new_path + 'test_lr.csv')
    lr_model.predict(new_path + 'lr_model.out', new_path + "test_pred.txt")

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
