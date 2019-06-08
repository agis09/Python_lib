# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from glob import glob
import tensorflow as tf
import keras
from keras import metrics, optimizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, ReLU, Flatten, Conv1D, MaxPooling1D
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from keras.callbacks import *
from tqdm import tqdm
import seaborn


class cnn_1D_doubleCV:
    def __init__(self, data, label, save_path):
        self.save_path = save_path
        self.data = data
        self.label = label
        self.n_split1 = 8
        self.n_split2 = 7
        # # # optuna_result # # #
        self.num_filters = [160, 144]
        self.drop_rate = 0.26347993763
        self.learning_rate = 0.0034726180431

    def make_model(self, input_shape, num_filters, drop_rate):
        in_layer = Input(input_shape)
        # layer = Flatten()(in_layer)

        layer = Conv1D(num_filters[0], 3)(in_layer)
        layer = BatchNormalization()(layer)
        layer = ReLU()(layer)
        layer = MaxPooling1D(2)(layer)

        layer = Conv1D(num_filters[1], 3)(layer)
        layer = BatchNormalization()(layer)
        layer = ReLU()(layer)
        layer = MaxPooling1D(2)(layer)

        layer = Flatten()(layer)
        layer = Dropout(drop_rate)(layer)
        layer = BatchNormalization()(layer)
        layer = Dense(3, activation='softmax')(layer)

        model = Model(in_layer, layer)
        return model

    def set_learning(self, train_data):
        model = make_model((train_data.shape[1], train_data.shape[2]),
                           self.num_filters, self.drop_rate)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=['accuracy'])
        return model

    def learning_curve_plot(self, result, save_path, cnt_1, cnt_2):

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].set_title('loss')
        ax[0].plot(result.epoch, result.history["loss"], label="Train loss")
        ax[0].plot(result.epoch, result.history["val_loss"],
                   label="Validation loss")
        ax[1].set_title('acc')
        ax[1].plot(result.epoch, result.history["acc"], label="Train acc")
        ax[1].plot(result.epoch, result.history["val_acc"],
                   label="Validation acc")
        ax[0].legend()
        ax[1].legend()

        plt.savefig(save_path+'result%d_%d.png' % (cnt_1, cnt_2))

    def heatmap_plot(self, heat_map, save_path):
        df_cm = pd.DataFrame(heat_map,
                             index=['true'+i for i in np.unique(self.labels)],
                             columns=['pred'+i for i in np.unique(self.labels)])
        plt.figure(figsize=(10, 7))
        seaborn.heatmap(df_cm, annot=True)
        plt.savefig(save_path+'heatmap.png')

    def exec_double_cv(self):
        skf = StratifiedKFold(n_splits=self.n_split1, shuffle=True)
        skf2 = StratifiedKFold(n_splits=self.n_split2, shuffle=True)

        label_num = len(np.unique(self.labels))
        heat_map = np.zeros((label_num, label_num))
        pred_list = np.array([])
        eval_list = np.array([])
        cnt_1 = 0
        for cv_train_idx, cv_eval_idx in tqdm(skf.split(data, labels)):

            x_cvtrain = data[cv_train_idx]
            y_cvtrain = labels[cv_train_idx]

            x_eval = data[cv_eval_idx]
            y_eval = labels[cv_eval_idx]

            cnt_2 = 0
            tmp_sum = np.zeros((label_num, label_num))

            for train_idx, test_idx in skf2.split(x_cvtrain, y_cvtrain):

                x_train = x_cvtrain[train_idx]
                y_train = y_cvtrain[train_idx]

                x_test = x_cvtrain[test_idx]
                y_test = y_cvtrain[test_idx]

                dict = {}
                for v, key in enumerate(np.unique(self.labels)):
                    dict.update({key: v})
                label_dict = dict

                y_train_tmp = [label_dict[lb] for lb in y_train]
                y_test_tmp = [label_dict[lb] for lb in y_test]
                y_eval_tmp = [label_dict[lb] for lb in y_eval]

                y_train = np.identity(label_num)[y_train_tmp]
                y_test = np.identity(label_num)[y_test_tmp]
                y_eval = np.identity(label_num)[y_eval_tmp]

                model = set_learning(x_train)
                cp_cb = ModelCheckpoint(filepath=self.save_path+'save_model%d_%d.hdf5' % (cnt_1, cnt_2),
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=True,
                                        mode='auto')

                hist = model.fit(x_train, y_train,
                                 batch_size=16,
                                 epochs=100,
                                 verbose=0,
                                 validation_data=(x_test, y_test),
                                 callbacks=[cp_cb])

                learning_curve_plot(hist, self.save_path, cnt_1, cnt_2)

                model.load_weights(
                    self.save_path+'save_model%d_%d.hdf5' % (cnt_1, cnt_2))
                # score_list.append(model.evaluate(x_eval, y_eval,verbose=0))

                y_pred = model.predict(x_eval)
                eval_list = np.append(eval_list, y_eval)
                pred_list = np.append(pred_list, y_pred)
                for i, j in zip(y_pred, y_eval):
                    tmp_sum[np.where(j == 1)] += i

                cnt_2 += 1
            heat_map += tmp_sum
            cnt_1 += 1

        eval_list = np.reshape(
            eval_list, (eval_list.shape[0]//label_num, label_num))
        pred_list = np.reshape(
            pred_list, (pred_list.shape[0]//label_num, label_num))

        eval_label = pd.DataFrame(eval_list)
        eval_label.to_csv(self.save_path+'eval_label.csv')
        pred_list = pd.DataFrame(pred_list)
        pred_list.to_csv(self.save_path+'pred_list.csv')

        heatmap_plot
