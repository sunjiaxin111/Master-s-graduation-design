# -*- coding: UTF-8 -*-
import time

import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    start = time.time()
    df = pd.read_csv('avazu.csv', sep='\t')
    print(df.shape)

    # 重命名
    df = df.rename(columns={'click': 'Label'})

    # 填充缺失值
    # 对类别类型的特征用众数填充
    cate_features = list(df.columns)
    cate_features.remove('id')
    cate_features.remove('Label')
    df[cate_features] = df[cate_features].fillna(df[cate_features].mode().iloc[0])

    # LabelEncoder
    le = LabelEncoder()
    for feat in cate_features:
        df[feat] = le.fit_transform(df[feat])

    # 查看每个类别特征的取值数,这里全部都是类别特征
    s = 0
    for i in cate_features:
        print(i, '的取值有', str(len(df[i].value_counts())), '个')
        s += len(df[i].value_counts())
    print("one-hot处理后的特征数为:", s)

    print(df.shape)
    features = list(df.columns)
    features.remove('id')
    features.remove('hour')
    df[features].to_csv('avazu_labelEncoder.csv', index=None, sep='\t')
    elapsed = (time.time() - start)
    print("Time used:", elapsed)
