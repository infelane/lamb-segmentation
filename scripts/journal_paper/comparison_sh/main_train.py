"""

Generate direct comparison to Shaoguangs paper:
- Small crop bottom right on 13
- Small crop bottom right on 19
"""

import matplotlib.pyplot as plt
import numpy as np

from scripts import standard_keras2

from datasets.default_trainingsets import get_13botleftshuang, get_19SE_shuang, get_1319, get_10lamb_old
from main_general import get_training_data
from methods.basic import NeuralNet
from methods.examples import compile_segm
from neuralNetwork.architectures import ti_unet, convNet, unet
from preprocessing.image import get_flow

# Settings (fixed)
model_name = 'ti-unet'  # ti-unet

"""
1319_10 (first trained on 1319, then on 10)
1319_101319 (first trained on 1319, then on 10, 13, 19)
'10' '1319' '13botright' '19botright'
"""

d = 2

w_patch = 10
if d == 1:
    w_ext_in = 10
elif d == 2:
    w_ext_in = 26

if model_name == 'unet':
    # lr = 1e-3
    # epochs = 1000   # Did not converge before...

    # It's probably just generally bad with low amount of n_per_class
    lr = 1e-4
    epochs = 100

else:
    lr = 1e-4
    epochs = 100

flow_tr = None


def load_data(data_name):

    def _data10nat():
        # Sparse annotations
        train_data = get_10lamb_old(5)

        from datasets.examples import get_10lamb
        from data.conversion_tools import annotations2y
        train_data.y_tr = annotations2y(get_10lamb().get("annot_tflearning"))

        return train_data

    if data_name == '13botright':
        train_data = get_13botleftshuang(5, n_per_class=n_per_class)

    elif data_name == '19botright':
        train_data = get_19SE_shuang(5, n_per_class=n_per_class)

    elif data_name == '1319':
        train_data = get_1319(5)

    elif data_name.split('_')[-1] == '10':

        # Sparse annotations
        train_data = get_10lamb_old(5)

    elif data_name.split('_')[-1] == '101319':
        from datasets.default_trainingsets import xy_from_df, get_13zach, get_19hand, get_10lamb, TrainData

        img_x10, img_y10 = xy_from_df(get_10lamb(), 5)
        img_x13, img_y13 = xy_from_df(get_13zach(), 5)
        img_x19, img_y19 = xy_from_df(get_19hand(), 5)

        img_x = [img_x10, img_x13, img_x19]
        img_y = [img_y10, img_y13, img_y19]

        # No test data
        train_data = TrainData(img_x, img_y, [np.zeros(shape=img_y_i.shape) for img_y_i in img_y])

    elif data_name.split('_')[-1] == '10nat':
        train_data = _data10nat()

    elif data_name.split('_')[-1] == '10nat1319':
        from datasets.default_trainingsets import TrainData

        train_data10 = _data10nat()
        train_data1319 = get_1319(5)

        img_x = [train_data10.get_x_train()] + train_data1319.get_x_train()
        img_y = [train_data10.get_y_train()] + train_data1319.get_y_train()

        train_data = TrainData(img_x,
                               img_y, [np.zeros(shape=img_y_i.shape) for img_y_i in img_y])

    else:
        raise ValueError(data_name)

    from data.preprocessing import rescale0to1
    train_data.x = rescale0to1(train_data.x)

    x_train, y_train, _, _ = get_training_data(train_data)

    global flow_tr
    flow_tr = get_flow(x_train, y_train,
                       w_patch=w_patch,
                       w_ext_in=w_ext_in
                       )

    return x_train, y_train


class Main(object):

    def __init__(self,
                 k=None
                 ):

        if k is not None:
            self.k = k

        self.model()

        self.train()

    def model(self, b_optimal_lr=False):

        if model_name == 'ti-unet':
            model = ti_unet(9, filters=self.k, w=w_patch, ext_in=w_ext_in // 2, batch_norm=True,
                            max_depth=d)

        elif model_name == 'unet':
            # model = ti_unet(9, filters=self.k, w=w_patch, ext_in=w_ext_in // 2, batch_norm=True,
            #                 max_depth=d)
            print('NO BATCH NORM? (not implemented)')
            model = unet(9, filters=self.k, w=w_patch, ext_in=w_ext_in // 2,
                         max_depth=d, n_per_block=1)

        else:
            raise ValueError(model_name)

        model.summary()

        compile_segm(model, lr=lr)

        if b_optimal_lr:
            from neuralNetwork.optimization import find_learning_rate

            global flow_tr
            find_learning_rate(model, flow_tr)

        self.neural_net = NeuralNet(model, w_ext=w_ext_in)

        if data_name[:5] == '1319_':    # pre Load model!
            # TODO which epoch to start from, I guess 10 should have an impact
            epoch_start = 50    # probably better (learned something)
            self.neural_net.load(f'C:/Users/admin/Data/ghent_altar/net_weight/1319/{model_name}_d{d}_k{self.k}', epoch=epoch_start)

    def train(self, epochs=epochs,
              steps_per_epoch=100
              ):

        global flow_tr

        info = f'{data_name}/{model_name}_d{d}_k{self.k}'

        try:
            if n_per_class is not None:
                info = '_'.join([info, f'n{n_per_class}'])
        except NameError:
            pass  # If n_per_class does not exist, don't add it


        self.neural_net.train(flow_tr,
                              # validation=flow_va,
                              epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              info=info)

        # Solve Memory problems?
        del self.neural_net

    def eval(self):
        pass


if __name__ == '__main__':

    if 0:
        x_train, y_train = load_data()

        n_per_class = 80
        for k in [9, 10, 11, 12]:    #[6]:  # range(1, 20)

            Main(k=k)

    elif 0:

        data_name = '19botright'

        # model_name = 'ti-unet'  # ti-unet
        # k = 9
        # for n_per_class in [640, 1280]:
        #
        #     x_train, y_train = load_data()
        #     Main(k=k)

        model_name = 'unet'  # ti-unet

        k = 9
        for n_per_class in [20]:
            x_train, y_train = load_data(data_name)
            Main(k=k)

    elif 0:

        n_per_class = None

        if 0:
            data_name = '1319_101319'  #

            for k in [8, 10, 12]:     # [6, 8, 10, 12]:    #[6]:  # range(1, 20)

                Main(k=k)

        for data_name in ['1319', '10', '1319_10', '1319_101319']:    # ,

            x_train, y_train = load_data(data_name)

            k = 10
            Main(k=k)

    elif 1:

        # On new annotations

        '10nat'    # "natural" annotations

        for data_name in ['1319_10nat', '10nat', '1319_10nat1319']:    # , '1319_10nat', '10nat'

            x_train, y_train = load_data(data_name)

            Main(k=10)
