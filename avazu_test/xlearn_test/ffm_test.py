# -*- coding: UTF-8 -*-
import time

import pandas as pd
import xlearn as xl
from sklearn import preprocessing, metrics


# 生成ffm格式
def convert_to_ffm(df, label_name, type, numerics, categories):
    currentcode = len(numerics)
    catdict = {}
    catcodes = {}
    # Flagging categorical and numerical fields
    for x in numerics:
        catdict[x] = 0
    for x in categories:
        catdict[x] = 1

    nrows = df.shape[0]
    with open(str(type) + "_ffm.txt", "w") as text_file:

        # Looping over rows to convert each row to libffm format
        for n, r in enumerate(range(nrows)):
            datastring = ""
            datarow = df.iloc[r].to_dict()
            datastring += str(int(datarow[label_name]))
            # For numerical fields, we are creating a dummy field here
            for i, x in enumerate(catdict.keys()):
                if (catdict[x] == 0):
                    datastring = datastring + " " + str(i) + ":" + str(i) + ":" + str(datarow[x])
                else:
                    # For a new field appearing in a training example
                    if (x not in catcodes):
                        catcodes[x] = {}
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode  # encoding the feature
                    # For already encoded fields
                    elif (datarow[x] not in catcodes[x]):
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode  # encoding the feature
                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " " + str(i) + ":" + str(int(code)) + ":1"

            datastring += '\n'
            text_file.write(datastring)


if __name__ == "__main__":
    start = time.time()
    df = pd.read_csv('../avazu_labelEncoder.csv', sep='\t')
    # 类别类型
    cate_features = list(df.columns)
    cate_features.remove("Label")

    # 连续型特征
    continuous_cols = []

    # 以6:2:2的比例分为训练集、验证集和测试集
    train = df[:int(df.shape[0] * 0.6)]
    val = df[int(df.shape[0] * 0.6):int(df.shape[0] * 0.8)]
    test = df[int(df.shape[0] * 0.8):]

    # 标准化
    for feat in continuous_cols:
        scaler = preprocessing.StandardScaler().fit(train[feat].values.reshape(-1, 1))
        train[feat] = scaler.transform(train[feat].values.reshape(-1, 1))
        val[feat] = scaler.transform(val[feat].values.reshape(-1, 1))
        test[feat] = scaler.transform(test[feat].values.reshape(-1, 1))

    df = pd.concat([train, val, test])

    # 先把FFM格式文件存到一个文件中，再分割
    convert_to_ffm(df, 'Label', 'avazu', continuous_cols, cate_features)

    # 得到Label值
    y_train = df[:int(df.shape[0] * 0.6)]['Label']
    y_valid = df[int(df.shape[0] * 0.6): int(df.shape[0] * 0.8)]['Label']
    y_test = df[int(df.shape[0] * 0.8):]['Label']

    df = pd.read_csv('avazu_ffm.txt', sep='\t\t\t\t', header=None)

    # 切分训练集、验证集、测试集,6:2:2
    train = df[:int(df.shape[0] * 0.6)]
    valid = df[int(df.shape[0] * 0.6):int(df.shape[0] * 0.8)]
    test = df[int(df.shape[0] * 0.8):]

    new_path = "FFM/"

    # 把训练集、验证集、测试集存储起来
    train.to_csv(new_path + 'train_ffm.csv', index=None, sep='\t', header=None)
    valid.to_csv(new_path + 'valid_ffm.csv', index=None, sep='\t', header=None)
    test.to_csv(new_path + 'test_ffm.csv', index=None, sep='\t', header=None)

    ffm_model = xl.create_ffm()
    ffm_model.setTrain(new_path + 'train_ffm.csv')
    ffm_model.setValidate(new_path + 'valid_ffm.csv')

    param = {'task': 'binary',
             'lr': 0.2,
             'lambda': 0.002,
             'metric': 'auc',
             'k': 8,
             'epoch': 999999
             }
    ffm_model.disableLockFree()
    ffm_model.fit(param, new_path + "ffm_model.out")

    # 预测
    ffm_model.setSigmoid()
    ffm_model.setTest(new_path + 'train_ffm.csv')
    ffm_model.predict(new_path + 'ffm_model.out', new_path + "train_pred.txt")
    ffm_model.setTest(new_path + 'valid_ffm.csv')
    ffm_model.predict(new_path + 'ffm_model.out', new_path + "valid_pred.txt")
    ffm_model.setTest(new_path + 'test_ffm.csv')
    ffm_model.predict(new_path + 'ffm_model.out', new_path + "test_pred.txt")

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
