# -*- coding: UTF-8 -*-
import logging
import time

import lightgbm as lgb
import pandas as pd

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

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_val, y_val)

### 设置初始参数--不含交叉验证参数
logging.warning('设置参数')
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.5,
    'max_depth': -1,
    'num_leaves': 80
}

### 交叉验证(调参)
logging.warning('交叉验证')
min_merror = float('Inf')
best_params = {}

# 过拟合
logging.warning("调参2：降低过拟合")
for max_bin in range(5, 255, 5):
    for min_data_in_leaf in range(10, 30, 5):
        params['max_bin'] = max_bin
        params['min_data_in_leaf'] = min_data_in_leaf

        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=42,
            nfold=3,
            metrics=['binary_logloss'],
            early_stopping_rounds=3,
            verbose_eval=True
        )

        mean_merror = pd.Series(cv_results['binary_logloss-mean']).min()
        boost_rounds = pd.Series(cv_results['binary_logloss-mean']).argmin()

        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params['max_bin'] = max_bin
            best_params['min_data_in_leaf'] = min_data_in_leaf

if 'min_data_in_leaf' in best_params:
    params['min_data_in_leaf'] = best_params['min_data_in_leaf']
if 'max_bin' in best_params:
    params['max_bin'] = best_params['max_bin']

logging.warning("调参3：降低过拟合")
for feature_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
    for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for bagging_freq in range(0, 50, 5):
            params['feature_fraction'] = feature_fraction
            params['bagging_fraction'] = bagging_fraction
            params['bagging_freq'] = bagging_freq

            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=42,
                nfold=3,
                metrics=['binary_logloss'],
                early_stopping_rounds=3,
                verbose_eval=True
            )

            mean_merror = pd.Series(cv_results['binary_logloss-mean']).min()
            boost_rounds = pd.Series(cv_results['binary_logloss-mean']).argmin()

            if mean_merror < min_merror:
                min_merror = mean_merror
                best_params['feature_fraction'] = feature_fraction
                best_params['bagging_fraction'] = bagging_fraction
                best_params['bagging_freq'] = bagging_freq

if 'feature_fraction' in best_params:
    params['feature_fraction'] = best_params['feature_fraction']
if 'bagging_fraction' in best_params:
    params['bagging_fraction'] = best_params['bagging_fraction']
if 'bagging_freq' in best_params:
    params['bagging_freq'] = best_params['bagging_freq']

logging.warning("调参4：降低过拟合")
for lambda_l1 in [0.0, 0.1, 0.2, 0.3, 0.4]:
    for lambda_l2 in [0.0, 0.1, 0.2, 0.3, 0.4]:
        for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4]:
            params['lambda_l1'] = lambda_l1
            params['lambda_l2'] = lambda_l2
            params['min_split_gain'] = min_split_gain

            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=42,
                nfold=3,
                metrics=['binary_logloss'],
                early_stopping_rounds=3,
                verbose_eval=True
            )

            mean_merror = pd.Series(cv_results['binary_logloss-mean']).min()
            boost_rounds = pd.Series(cv_results['binary_logloss-mean']).argmin()

            if mean_merror < min_merror:
                min_merror = mean_merror
                best_params['lambda_l1'] = lambda_l1
                best_params['lambda_l2'] = lambda_l2
                best_params['min_split_gain'] = min_split_gain

if 'lambda_l1' in best_params:
    params['lambda_l1'] = best_params['lambda_l1']
if 'lambda_l2' in best_params:
    params['lambda_l2'] = best_params['lambda_l2']
if 'min_split_gain' in best_params:
    params['min_split_gain'] = best_params['min_split_gain']

logging.warning(params)

elapsed = (time.time() - start)
print("Time used:", elapsed)
