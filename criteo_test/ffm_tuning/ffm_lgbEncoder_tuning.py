# -*- coding: UTF-8 -*-
import argparse
import ast
import logging
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, log_loss
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


class FeatureDictionary(object):
    def __init__(self, trainfile=None, testfile=None,
                 dfTrain=None, dfTest=None, numeric_cols=[], ignore_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"
        self.trainfile = trainfile
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)
        else:
            dfTrain = self.dfTrain
        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)
        else:
            dfTest = self.dfTest
        df = pd.concat([dfTrain, dfTest])
        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                # map to a single index
                self.feat_dict[col] = tc
                tc += 1
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)
        self.feat_dim = tc


class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)
        y = dfi["Label"].values.tolist()
        dfi.drop(["Label"], axis=1, inplace=True)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        # list of list of feature indices of each sample in the dataset
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        Xv = dfv.values.tolist()

        return Xi, Xv, y


class DeepAFFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size,
                 embedding_size=8, attention_size=10, dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 use_ffm=True, use_deep=True, use_attention=True,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True):
        assert (use_ffm or use_deep)
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size  # denote as M, size of the feature dictionary
        self.field_size = field_size  # denote as F, size of the feature fields
        self.embedding_size = embedding_size  # denote as K, size of the feature embedding
        self.attention_size = attention_size

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.use_ffm = use_ffm
        self.use_deep = use_deep
        self.use_attention = use_attention
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        # 直接把测试集的结果也保存下来，就可以直接取出valid上最优的结果了
        self.train_auc, self.valid_auc, self.test_auc = [], [], []
        self.train_logloss, self.valid_logloss, self.test_logloss = [], [], []

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()  # 新建图
        with self.graph.as_default():  # 把该图作为默认图

            tf.set_random_seed(self.random_seed)  # 设置随机数种子
            np.random.seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name="feat_index")  # 使用样本数*field_size
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name="feat_value")  # 使用样本数*field_size
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # 使用样本数 * 1
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")  # fm层dropout保留的比例
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None],
                                                    name="dropout_keep_deep")  # deep层dropout保留的比例
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")  # 是否是训练阶段的flag

            self.weights = self._initialize_weights()  # 初始化权重，即变量

            # model
            # embedding查表
            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                     self.feat_index)  # 使用样本数*field_size*field_size*embedding_size
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings,
                                          tf.reshape(self.feat_value, shape=[-1, self.field_size, 1, 1]))

            # ---------- first order term ----------
            self.y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"],
                                                        self.feat_index)  # None * field_size * 1
            # reduce_sum函数用于在某一维度求和
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)  # None * F
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0])  # None * F

            # ---------- element_wise ---------------
            element_wise_product_list = []
            for i in range(self.field_size):
                for j in range(i + 1, self.field_size):
                    element_wise_product_list.append(
                        tf.multiply(self.embeddings[:, i, j, :], self.embeddings[:, j, i, :]))  # None * K

            self.element_wise_product = tf.stack(element_wise_product_list)  # (F * F - 1 / 2) * None * K
            self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1, 0, 2],
                                                     name='element_wise_product')  # None * (F * F - 1 / 2) *  K
            self.element_wise_product = tf.nn.dropout(self.element_wise_product, self.dropout_keep_fm[1])  # None * K

            if self.use_attention:
                # attention part
                num_interactions = int(self.field_size * (self.field_size - 1) / 2)
                # wx+b -> relu(wx+b) -> h*relu(wx+b)
                self.attention_wx_plus_b = tf.reshape(
                    tf.add(tf.matmul(tf.reshape(self.element_wise_product, shape=(-1, self.embedding_size)),
                                     self.weights['attention_w']),
                           self.weights['attention_b']),
                    shape=[-1, num_interactions, self.attention_size])  # N * ( F * F - 1 / 2) * A

                self.attention_exp = tf.exp(tf.reduce_sum(tf.multiply(tf.nn.relu(self.attention_wx_plus_b),
                                                                      self.weights['attention_h']),
                                                          axis=2, keep_dims=True))  # N * ( F * F - 1 / 2) * 1

                self.attention_exp_sum = tf.reduce_sum(self.attention_exp, axis=1, keep_dims=True)  # N * 1 * 1

                self.attention_out = tf.div(self.attention_exp, self.attention_exp_sum,
                                            name='attention_out')  # N * ( F * F - 1 / 2) * 1

                self.attention_x_product = tf.reduce_sum(tf.multiply(self.attention_out, self.element_wise_product),
                                                         axis=1,
                                                         name='afm')  # N * K

                self.second_order_part_sum = tf.matmul(self.attention_x_product, self.weights['attention_p'])  # N * 1
            else:
                self.second_order_part_sum = tf.reshape(tf.reduce_sum(self.element_wise_product, axis=[1, 2]),
                                                        shape=[-1, 1])

            # ---------- Deep component ----------
            self.y_deep = tf.reshape(self.embeddings, shape=[-1,
                                                             self.field_size * self.field_size * self.embedding_size])  # None * (F*F*K)
            # self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])  # 这一行存疑，在输入层是否需要dropout
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]),
                                     self.weights["bias_%d" % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase,
                                                        scope_bn="bn_%d" % i)  # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1 + i])  # dropout at each Deep layer

            # ---------- DeepFFM ----------
            if self.use_ffm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.second_order_part_sum, self.y_deep], axis=1)
            elif self.use_ffm:
                concat_input = tf.concat([self.y_first_order, self.second_order_part_sum], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep
            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d" % i])

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                logging.warning("#params: %d" % total_parameters)

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _initialize_weights(self):
        weights = dict()

        # embedding权重矩阵
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.field_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_size * field_size * embedding_size
        # wx+b中的w
        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1

        # attention part
        if self.use_attention:
            glorot = np.sqrt(2.0 / (self.attention_size + self.embedding_size))

            weights['attention_w'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.embedding_size, self.attention_size)),
                dtype=tf.float32, name='attention_w')

            weights['attention_b'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.attention_size,)),
                                                 dtype=tf.float32, name='attention_b')

            weights['attention_h'] = tf.Variable(np.random.normal(loc=0, scale=1, size=(self.attention_size,)),
                                                 dtype=tf.float32, name='attention_h')

            weights['attention_p'] = tf.Variable(np.ones((self.embedding_size, 1)), dtype=np.float32,
                                                 name='attention_p')

        # deep layers
        num_layer = len(self.deep_layers)  # 隐藏层层数
        input_size = self.field_size * self.field_size * self.embedding_size  # dnn的输入
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))  # 计算后续用到的标准差
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        if self.use_ffm and self.use_deep:
            input_size = self.field_size + 1 + self.deep_layers[-1]
        elif self.use_ffm:
            input_size = self.field_size + 1
        elif self.use_deep:
            input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
            dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)  # wx+b中的b

        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously  这里的打乱是随机的，会导致每次执行程序的结果不同
    def shuffle_in_unison_scary(self, a, b, c):
        np.random.seed(2019)
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None, Xi_test=None, Xv_test=None, y_test=None,
            early_stopping=False):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time.time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)

            # evaluate training and validation datasets
            train_auc = self.evaluate(Xi_train, Xv_train, y_train, roc_auc_score)
            self.train_auc.append(train_auc)
            train_logloss = self.evaluate(Xi_train, Xv_train, y_train, log_loss)
            self.train_logloss.append(train_logloss)
            if has_valid:
                valid_auc = self.evaluate(Xi_valid, Xv_valid, y_valid, roc_auc_score)
                self.valid_auc.append(valid_auc)
                valid_logloss = self.evaluate(Xi_valid, Xv_valid, y_valid, log_loss)
                self.valid_logloss.append(valid_logloss)

                test_auc = self.evaluate(Xi_test, Xv_test, y_test, roc_auc_score)
                self.test_auc.append(test_auc)
                test_logloss = self.evaluate(Xi_test, Xv_test, y_test, log_loss)
                self.test_logloss.append(test_logloss)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    logging.warning("[%d] train-auc=%.4f, valid-auc=%.4f [%.1f s]"
                                    % (epoch + 1, train_auc, valid_auc, time.time() - t1))
                else:
                    logging.warning("[%d] train-auc=%.4f [%.1f s]"
                                    % (epoch + 1, train_auc, time.time() - t1))
            if has_valid and early_stopping and self.training_termination(self.valid_auc):
                break

    def training_termination(self, valid_result):
        # 这里应该是>=，因为有可能从第一个epoch就过拟合了
        if len(valid_result) >= 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                        valid_result[-2] < valid_result[-3] and \
                        valid_result[-3] < valid_result[-4] and \
                        valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                        valid_result[-2] > valid_result[-3] and \
                        valid_result[-3] > valid_result[-4] and \
                        valid_result[-4] > valid_result[-5]:
                    return True
        return False

    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred

    def evaluate(self, Xi, Xv, y, eval_metric):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv)
        return eval_metric(y, y_pred)


def _load_data(config):
    dfTrain = pd.read_csv(config['TRAIN_FILE'], sep='\t')
    dfTest = pd.read_csv(config['TEST_FILE'], sep='\t')

    cols = [c for c in dfTrain.columns if c not in config['IGNORE_COLS']]

    X_train = dfTrain[cols].values
    y_train = dfTrain["Label"].values
    X_test = dfTest[cols].values
    y_test = dfTest["Label"].values
    # 生成类别特征列表
    cate_list = []
    for feat in cols:
        if feat.startswith('leaf'):
            cate_list.append(feat)
    cat_features_indices = [i for i, c in enumerate(cols) if c in cate_list]

    return dfTrain, dfTest, X_train, y_train, X_test, y_test, cat_features_indices


def _run_base_model_dfm(dfTrain, dfTest, dfm_params, config):
    # 生成数值特征列表和忽略列表
    numeric_list = []
    ignore_list = ['Label']
    features = list(dfTrain.columns)
    features.remove("Label")
    for feat in features:
        if feat.startswith('cross_entropy'):
            if config['use_cross_entropy']:
                numeric_list.append(feat)
            else:
                ignore_list.append(feat)
        elif feat.startswith('pos_ratio'):
            if config['use_pos_ratio']:
                numeric_list.append(feat)
            else:
                ignore_list.append(feat)

    # 这里默认除了数值型就是类别型，所以如果没有使用交叉熵信息或者正例率特征，需要特殊处理
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=numeric_list,
                           ignore_cols=ignore_list)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain)
    Xi_test, Xv_test, y_test = data_parser.parse(df=dfTest)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    # 从train中取25%为valid
    Xi_train_ = Xi_train[:int(len(Xi_train) * 0.75)]
    Xv_train_ = Xv_train[:int(len(Xv_train) * 0.75)]
    y_train_ = y_train[:int(len(y_train) * 0.75)]
    Xi_valid_ = Xi_train[int(len(Xi_train) * 0.75):]
    Xv_valid_ = Xv_train[int(len(Xv_train) * 0.75):]
    y_valid_ = y_train[int(len(y_train) * 0.75):]

    dfm = DeepAFFM(**dfm_params)
    dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_, Xi_test, Xv_test, y_test,
            early_stopping=True)

    # 看效果应在函数里面看，因为y_train_这些变量在训练时被打乱了
    # 训练集效果
    logging.warning("训练集logloss: %.8f" % dfm.train_logloss[-5])
    logging.warning("训练集auc: %.8f" % dfm.train_auc[-5])
    # 验证集效果
    logging.warning("验证集logloss: %.8f" % dfm.valid_logloss[-5])
    logging.warning("验证集auc: %.8f" % dfm.valid_auc[-5])
    # 测试集效果
    logging.warning("测试集logloss: %.8f" % dfm.test_logloss[-5])
    logging.warning("测试集auc: %.8f" % dfm.test_auc[-5])

    return dfm.valid_auc[-5], dfm.valid_logloss[-5]


def parse_args():
    parser = argparse.ArgumentParser(description="Run deepAFFM.")
    parser.add_argument('--use_cross_entropy', type=ast.literal_eval, default=True,
                        help='use_cross_entropy type: True, False.')
    parser.add_argument('--use_pos_ratio', type=ast.literal_eval, default=True,
                        help='use_cross_entropy type: True, False.')
    parser.add_argument('--embedding_size_list', nargs='?', default='[2,4]',
                        help='embedding_size_list')
    parser.add_argument('--dropout_fm_0_list', nargs='?', default='[0.5,0.7,0.9]',
                        help='dropout_fm_0_list')
    parser.add_argument('--dropout_fm_1_list', nargs='?', default='[0.5,0.7,0.9]',
                        help='dropout_fm_1_list')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.use_cross_entropy:
        logging.warning("使用交叉熵特征")
    if args.use_pos_ratio:
        logging.warning("使用正例率特征")

    start = time.time()
    df = pd.read_csv('../criteo_lgbEncoder.csv', sep='\t')

    new_path = "data/"
    df[:int(df.shape[0] * 0.8)].to_csv(new_path + "train.csv", index=None, sep='\t')
    df[int(df.shape[0] * 0.8):].to_csv(new_path + "test.csv", index=None, sep='\t')

    # 原config信息
    # set the path-to-files
    config = {}
    config['TRAIN_FILE'] = "./data/train.csv"
    config['TEST_FILE'] = "./data/test.csv"
    config['RANDOM_SEED'] = 2019
    config['IGNORE_COLS'] = ["Label"]
    config['use_cross_entropy'] = args.use_cross_entropy
    config['use_pos_ratio'] = args.use_pos_ratio

    # load data
    dfTrain, dfTest, X_train, y_train, X_test, y_test, cat_features_indices = _load_data(config)

    # ------------------ FFM Model ------------------
    # params
    params = {
        "use_ffm": True,
        "use_deep": False,
        "use_attention": False,
        "attention_size": 10,
        "embedding_size": 8,
        "dropout_fm": [1.0, 1.0],
        "deep_layers": [16, 16],
        "dropout_deep": [0.5, 0.5, 0.5],
        "deep_layers_activation": tf.nn.relu,
        "epoch": 9999999,
        "batch_size": 256,
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.01,
        "verbose": True,
        "random_seed": config['RANDOM_SEED']
    }

    best_params = {}

    # ------------------ FFM Model tuning------------------
    logging.warning("FFM")
    # (调参)
    logging.warning('调参')
    # 设置双重条件，起始的auc和logloss设置为使用lgb时的效果
    max_valid_auc = float('-Inf')
    min_valid_logloss = float('Inf')

    logging.warning('调attention_size')
    for embedding_size in eval(args.embedding_size_list):
        for dropout_fm_1 in eval(args.dropout_fm_0_list):
            for dropout_fm_2 in eval(args.dropout_fm_1_list):
                logging.warning("embedding_size: %d" % embedding_size)
                logging.warning("dropout_fm:[%.1f, %.1f]" % (dropout_fm_1, dropout_fm_2))
                params['embedding_size'] = embedding_size
                params['dropout_fm'] = [dropout_fm_1, dropout_fm_2]
                valid_auc, valid_logloss = _run_base_model_dfm(dfTrain, dfTest, params, config)

                if max_valid_auc < valid_auc and min_valid_logloss > valid_logloss:
                    max_valid_auc = valid_auc
                    min_valid_logloss = valid_logloss
                    best_params['embedding_size'] = embedding_size
                    best_params['dropout_fm'] = [dropout_fm_1, dropout_fm_2]

    if 'embedding_size' in best_params:
        params['embedding_size'] = best_params['embedding_size']
        logging.warning("best embedding_size: %d" % best_params['embedding_size'])

    if 'dropout_fm' in best_params:
        params['dropout_fm'] = best_params['dropout_fm']
        logging.warning("best dropout_fm:[%.1f, %.1f]" % (best_params['dropout_fm'][0], best_params['dropout_fm'][1]))

    # 调完参数后，再运行一遍模型得到最后的效果
    logging.warning("调参结束！")
    logging.warning(params)
    _run_base_model_dfm(dfTrain, dfTest, params, config)

    elapsed = (time.time() - start)
    print("Time used:", elapsed)
