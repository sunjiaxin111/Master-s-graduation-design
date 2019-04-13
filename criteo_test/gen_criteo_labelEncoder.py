# -*- coding: UTF-8 -*-
import time

import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    start = time.time()
    df = pd.read_csv('criteo.csv', sep='\t')
    print(df.shape)

    # 填充缺失值
    # 对类别类型的特征用众数填充
    cate_features = ['C1', 'C2', 'C3', 'C4', 'C5',
                     'C6', 'C7', 'C8', 'C9', 'C10',
                     'C11', 'C12', 'C13', 'C14', 'C15',
                     'C16', 'C17', 'C18', 'C19', 'C20',
                     'C21', 'C22', 'C23', 'C24', 'C25',
                     'C26']
    df[cate_features] = df[cate_features].fillna(df[cate_features].mode().iloc[0])
    # 对数值类型的特征用中位数填充
    number_features = ['I1', 'I2', 'I3', 'I4', 'I5',
                       'I6', 'I7', 'I8', 'I9', 'I10',
                       'I11', 'I12', 'I13']
    df[number_features] = df[number_features].fillna(df[number_features].median())

    # LabelEncoder
    le = LabelEncoder()
    for feat in cate_features:
        df[feat] = le.fit_transform(df[feat])

    # 查看每个类别特征的取值数
    s = 0
    for i in cate_features:
        print(i, '的取值有', str(len(df[i].value_counts())), '个')
        s += len(df[i].value_counts())
    # 加上数值型特征个数
    s += len(number_features)
    print("one-hot处理后的特征数为:", s)

    print(df.shape)
    df.to_csv('criteo_labelEncoder.csv', index=None, sep='\t')
    elapsed = (time.time() - start)
    print("Time used:", elapsed)
