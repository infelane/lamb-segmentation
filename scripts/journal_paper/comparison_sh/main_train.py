"""

Generate direct comparison to Shaoguangs paper:
- Small crop bottom right on 13
- Small crop bottom right on 19
"""

import matplotlib.pyplot as plt
import numpy as np

from scripts import standard_keras2

from methods.basic import NeuralNet
from methods.examples import compile_segm
from neuralNetwork.architectures import ti_unet, convNet, unet
from preprocessing.image import get_flow
from main_general import get_training_data

from scripts.journal_paper.comparison_sh.shared import load_data

# Settings (fixed)
model_name = 'ti-unet'  # ti-unet

"""
1319_10 (first trained on 1319, then on 10)
1319_101319 (first trained on 1319, then on 10, 13, 19)
'10' '1319' '13botright' '19botright'
"""

d = 1

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


class Main(object):

    def __init__(self,
                 k=None,
                 seed=None
                 ):

        if k is not None:
            self.k = k
        
        self.seed = seed

        self.model()

        self.train()

    def model(self, b_optimal_lr=False):
    
        features_out = 3 if data_name == '19botrightcrack3' else 2

        if model_name == 'ti-unet':
            model = ti_unet(9, filters=self.k, w=w_patch, ext_in=w_ext_in // 2, batch_norm=True,
                            max_depth=d,
                            features_out=features_out,
                            )

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
        
        if self.seed is not None:
            info += f'_s{self.seed}'

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
    
    def set_flow(train_data):
    
        x_train, y_train, _, _ = get_training_data(train_data)
        
        global flow_tr
        flow_tr = get_flow(x_train, y_train,
                           w_patch=w_patch,
                           w_ext_in=w_ext_in
                           )

    if 0:
        x_train, y_train = load_data()

        n_per_class = 80
        for k in [9, 10, 11, 12]:    #[6]:  # range(1, 20)

            Main(k=k)

    elif 0:
        
        data_name = '1319botright'
        # data_name = '19botrightcrack3'   # '19botright, '19botrightcrack', '19botrightcrack3'

        model_name = 'ti-unet'  # ti-unet

        k = 9
        for n_per_class in [80, 160, 320, 640, 1280]:
            train_data = load_data(data_name, n_per_class)
            set_flow(train_data)
            Main(k=k)

    elif 1:
        """
        Multiple iterations for graph 10
        """
        
        data_name = '19botright'
        k = 9

        for model_name in ['ti-unet', 'unet']:
            for n_per_class in [80, 160, 320, 640, 1280, 20, 40]:
                for seed in [100, 200, 300]:
                    train_data = load_data(data_name, n_per_class, seed=seed)
                    set_flow(train_data)
                    Main(k=k, seed=seed)
    
    elif 0:

        n_per_class = None

        if 0:
            data_name = '1319_101319'  #

            for k in [8, 10, 12]:     # [6, 8, 10, 12]:    #[6]:  # range(1, 20)

                Main(k=k)

        for data_name in ['1319', '10', '1319_10', '1319_101319']:    # ,

            train_data = load_data(data_name, n_per_class)
            set_flow(train_data)

            k = 10
            Main(k=k)

    elif 1:

        # On new annotations

        '10nat'    # "natural" annotations

        for data_name in ['1319_10nat', '10nat', '1319_10nat1319']:    # , '1319_10nat', '10nat'

            train_data = load_data(data_name, n_per_class)
            set_flow(train_data)

            Main(k=10)
