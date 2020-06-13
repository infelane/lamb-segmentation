import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

from data.datatools import pandas_save
from plotting import concurrent
from performance.testing import test_thresh_incremental, test_thresh_optimal

# Settings:

mod = 5


class Main(object):

    k = 8
    n_per_class = 80
    d = 1

    epochs = 100

    if d == 1:
        w_ext_in = 10
    elif d == 2:
        w_ext_in = 26

    w_patch = 10

    if 0:
        lr = 1e-3
        steps_per_epoch = 100
    else:
        lr = 1e-4
        steps_per_epoch = 1000

    def __init__(self,
                 k=None,
                 n_per_class=None
                 ):

        if k is not None:
            self.k = k

        if n_per_class is not None:
            self.n_per_class = n_per_class

        # Get net
        self.model_train = self.main_net(set_net)
        from methods.basic import NeuralNet

        self.neural_net = NeuralNet(self.model_train, w_ext=self.w_ext_in)

        # Get data
        train_data = self.main_data(set_data)

        n_val_datas = len(self.val_datas)
        lst_data = [[] for _ in range(n_val_datas + 1)]
        for _ in range(self.epochs):
            # Train
            self.main_train(train_data)

            # Evaluate
            data_lst = self.main_eval(train_data)

            for i, data_i in enumerate(data_lst):
                data_i.update({'epoch': self.neural_net.epoch})

                lst_data[i].append(data_i)

        for i in range(n_val_datas + 1):
            df = pd.DataFrame(lst_data[i])
            print(df)

            if i == 0:
                data_name = 'val'
            else:
                data_name = self.val_datas[i-1]['name']
            model_name = f'{set_net["name"]}_data{data_name}_d{self.d}_k{self.k}_n{self.n_per_class}'
            pandas_save(f'C:/Users/admin/OneDrive - ugentbe/data/dataframes/{model_name}.csv', df, append=True)

        if 0:
            if 0:
                self.neural_net.load('C:/Users/admin/Data/ghent_altar/net_weight/tiunet_d1_k10_n80', 4)
            self.main_eval(train_data, b_plot=True)

        print("Finished init")

    def main_net(self, set_n):
        from methods.examples import compile_segm
        from neuralNetwork.architectures import ti_unet, convNet, unet

        n_name = set_n['name'].lower()
        if n_name == 'ti-unet':
            model = ti_unet(9, filters=self.k, w=self.w_patch, ext_in=self.w_ext_in // 2, batch_norm=True,
                                   max_depth=self.d)
        elif n_name == 'simple':
            model = convNet(9, self.k, w_in=self.w_patch+self.w_ext_in, n_convs=5,
                            batch_norm=False, padding='valid')

            assert model.output_shape[-3:] == (self.w_patch, self.w_patch, 2)
            
        elif n_name == 'unet':
            print('NO BATCH NORM? (not implemented)')
            model = unet(9, filters=self.k, w=self.w_patch, ext_in=self.w_ext_in // 2,
                                   max_depth=self.d, n_per_block=1)
        else:
            raise ValueError(n_name)

            # raise NotImplementedError('Unet is not well implemented: * Double, batchnorm? f per layer etc?')

        model.summary()
        compile_segm(model, lr=self.lr)     # instead of 10e-3, 10e-4 is probs more stable.

        return model

    def main_data(self, set_data):

        if set_data['name'] == 'zach_sh':
            from datasets.default_trainingsets import get_13botleftshuang
            train_data = get_13botleftshuang(mod, n_per_class=self.n_per_class)
        else:
            raise NotImplementedError()

        from data.preprocessing import rescale0to1
        train_data.x = rescale0to1(train_data.x)

        from datasets.default_trainingsets import xy_from_df, panel13withoutRightBot
        from datasets.examples import get_13zach

        _, img_y = xy_from_df(get_13zach(), mod)
        img_y_top2, _ = panel13withoutRightBot(img_y)

        img_y_test = np.logical_or(img_y_top2, train_data.get_y_test())

        self.val_datas = [{'name': '13_top2',
                           'y': img_y_top2},
                          {'name': '13_test',
                           'y': img_y_test
                           }
                          ]

        return train_data

    def main_train(self, train_data,
                   steps_per_epoch=None
                   ):
        if steps_per_epoch is None:
            steps_per_epoch = self.steps_per_epoch

        from main_general import get_training_data

        from preprocessing.image import get_flow

        # TODO train
        x_train, y_train, x_val, y_val = get_training_data(train_data)

        # Generator
        flow_tr = get_flow(x_train, y_train,
                           w_patch=self.w_patch,
                           w_ext_in=self.w_ext_in
                           )

        flow_va = get_flow(x_val, y_val,
                           w_patch=self.w_patch,
                           w_ext_in=self.w_ext_in
                           )

        epochs = 1

        self.neural_net.train(flow_tr,
                              validation=flow_va,
                              epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              info=f'{set_net["name"]}_d{self.d}_k{self.k}_n{self.n_per_class}')

    def main_eval(self, train_data,
                  b_plot=False,
                  ):

        x = train_data.get_x_test()

        y_pred = self.neural_net.predict(x)

        if b_plot:
            concurrent([x[..., :3], y_pred[..., 1]], ['input', 'prediction'])

        val_datas = [{'y': train_data.get_y_test()}] + self.val_datas

        return _eval_func(y_pred, val_datas, b_plot=b_plot)


def _eval_func(y_pred, val_datas):

    data_lst = []

    for val_data in val_datas:
        y_true = val_data['y']

        data_i = _eval_func_single(y_true, y_pred)

        data_lst.append(data_i)

    return data_lst


def _eval_func_single(y_true, y_pred, metric='kappa'):
    # calculate the threshold

    # t = test_thresh_incremental(y_pred, y_te=y_true, n=5)
    t = test_thresh_optimal(y_pred, y_true, metric=metric)

    acc, jacc, kappa = eval_scores(y_true, y_pred, t=t)

    data_i = {'thresh': t,
              'kappa': kappa,
              'accuracy': acc,
              'jaccard': jacc
              }

    return data_i


def eval_scores(y_true, y_pred, t=.5):
    from performance.testing import _get_scores , filter_non_zero, get_y_pred_thresh

    y_pred_thresh = get_y_pred_thresh(y_pred, t)

    # calculate scores
    y_test_arg, y_pred_arg = map(lambda y: np.argmax(y, axis=-1), filter_non_zero(y_true, y_pred_thresh))

    acc, jacc, kappa = _get_scores(y_test_arg, y_pred_arg)

    return acc, jacc, kappa


if __name__ == '__main__':

    set_data = {'name': 'zach_sh',
                }

    for k in [9, 11]:    # [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        for n_per_class in [80, 160, 320]:

            for name in ['unet', 'ti-unet', 'simple']:      # name in ['simple', 'ti-unet', 'u-net']
                set_net = {'name': name,
                           }

                m = Main(k=k, n_per_class=n_per_class)
                del m
