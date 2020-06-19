import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts import standard_keras2

from methods.basic import NeuralNet
from plotting import concurrent
from scripts.scripts_performance.main_performance import load_model_quick
from scripts.main_simple import _eval_func_single
from main_general import get_training_data

from data.datatools import pandas_save

# n_per_class = None    # 80, None

data_train = '19botright'     # '19botrightcrack' '13botright', '19botright', '10', '1319', '1319_10'

data_eval = None    # '10' '19botright'  None

metric = 'kappa'

if data_eval is None:
    data_eval = data_train

model_name = 'ti-unet'   # 'ti-unet', 'unet'

d = 1

set_eval = 'n_per_class'    # 'k', 'n_per_class'


def gen_k():
    for k in range(1, 16):
        if n_per_class is None:
            yield {'k': k, 'n_per_class': None}  # some weird bug here
        else:
            yield {'k': k, 'n_per_class': n_per_class}


def gen_n_per_class():

    k = 9

    for n_per_class in [1280, 640, 320, 160, 80, 40, 20, 10]:  # [10, 20, 40, 80, 160, 320, 640, 1280]
        
        for s in [None, 100, 200, 300]:
            yield {'k': k,
                   'n_per_class': n_per_class,
                   'seed':s}

folder_base = 'C:/Users/admin/Data/ghent_altar/' if os.name == 'nt' else '/scratch/lameeus/data/ghent_altar/'



class Main(object):

    def __init__(self):

        self.l = []

        if set_eval == 'k':

            gen = gen_k
            self.load_data()

        elif set_eval == 'n_per_class':

            assert '19botright' in data_train, data_train

            gen = gen_n_per_class

        else:
            raise ValueError(set_eval)

        for di in gen():

            k, n_per_class = di['k'], di['n_per_class']
            seed = di['seed']
            print('k:', k, '; n_per_class:', n_per_class)
            
            self.load_data(n_per_class, seed=seed)
            
            foldername = f'net_weight/{data_train}/{model_name}_d{d}_k{k}'
            
            folder = os.path.join(folder_base, foldername)
            if n_per_class is not None:
                folder += f'_n{n_per_class}'
            if seed is not None:
                folder += f'_s{seed}'

            if not os.path.exists(folder):
                continue

            epochs = []
            for w_file in os.listdir(folder):
                epoch = int(os.path.splitext(w_file)[0].split('_')[1])
                epochs.append(epoch)
            epochs.sort()

            def m(e, metric='kappa'):
                print('epoch:', e)

                self.load_model(epoch=e, k=k, n_per_class=n_per_class, seed=seed)

                data_i = self.eval()

                data_i['k'] = k
                data_i['n_per_class'] = n_per_class
                data_i['epoch'] = e

                if seed is not None:
                    data_i['seed'] = seed

                self.l.append(data_i)

                if metric == 'kappa':
                    return data_i['kappa']
                else:
                    raise ValueError(metric)

            def delta_min(e0, emid, e1):
                # return max(emid - e0, e1 - emid)
                return min(emid - e0, e1 - emid)    # TODO stop a bit later?

            def round(x):
                return int(np.round(x))

            # Find best epoch
            e0_old = epochs[0]
            e_best = epochs[len(epochs) // 2 - 1]
            e1_old = epochs[-1]

            v0_old = m(e0_old)
            v1_old = m(e1_old)
            v_best = m(e_best)

            delta = delta_min(e0_old, e_best, e1_old)

            while delta >= 2:

                e0_new = round((e0_old + e_best) / 2.)
                e1_new = round((e_best + e1_old) / 2.)

                v0_new = m(e0_new)
                v1_new = m(e1_new)

                e_lst = [e0_old, e0_new, e_best, e1_new, e1_old]
                v_lst = [v0_old, v0_new, v_best, v1_new, v1_old]

                # Update crop
                i_max, v_best = max(enumerate(v_lst), key=lambda x: x[1])

                if i_max == len(e_lst) - 1:    # Last element is biggest
                    i_max += -1  # Just act as if index before is best.

                elif i_max == 0:    # First element is biggest
                    i_max += 1  # Just act as if index after is best.

                e_best = e_lst[i_max]

                e0_old = e_lst[i_max - 1]
                e1_old = e_lst[i_max + 1]

                v0_old = v_lst[i_max - 1]
                v1_old = v_lst[i_max + 1]

                delta_temp = delta_min(e0_old, e_best, e1_old)
                if delta_temp >= delta:
                    break
                delta = delta_temp

            self.safe()

        self.safe(plot=True)

        print('init done')

    def load_model(self,
                   epoch,
                   k,
                   n_per_class,
                   seed = None
                   ):

        l_info = [model_name, 'd1', f'k{k}',  ]
        if n_per_class is not None:
            l_info.append(f"n{n_per_class}")
            
        if seed is not None:
            l_info.append(f"s{seed}")
            
        path = os.path.join(folder_base, f'net_weight/{data_train}/{"_".join(l_info)}/w_{epoch}.h5')

        model = load_model_quick(path)
        neural_net = NeuralNet(model, w_ext=10, norm_x=True)

        self.neural_net = neural_net

    def load_data(self, n_per_class=None, seed=None):
        from scripts.journal_paper.comparison_sh.shared import load_data as load_data_shared

        if 0:

            if data_eval == '13botright':
                from datasets.default_trainingsets import xy_from_df, get_13zach, panel13withoutRightBot
                img_x, img_y = xy_from_df(get_13zach(), 5)
    
                _, y_val = panel13withoutRightBot(img_y)
    
                if set_eval == 'n_per_class':
                    raise ValueError('n_per_class should not be evaluated on 13 right bot! Is too small!!')
    
            elif data_eval == '19botright':
                from datasets.default_trainingsets import xy_from_df, get_19hand, panel19withoutRightBot
                img_x, img_y = xy_from_df(get_19hand(), 5)
    
                if set_eval == 'n_per_class':
                    from datasets.default_trainingsets import get_19SE_shuang
                    td = get_19SE_shuang(5, n_per_class=320)
                    y_val = td.get_y_test() # Probably most proper way to do it (does cause the performance to be slightly sub, but should be fine)
    
                else:
                    _, y_val = panel19withoutRightBot(img_y)
    
            elif data_eval == '10':
                from datasets.examples import get_10lamb, get_10lamb_kfold
                from datasets.default_trainingsets import xy_from_df, get_19hand, panel19withoutRightBot
    
                img_x, _ = xy_from_df(get_10lamb(), 5)
                y_val = get_10lamb_kfold().get("annot_clean_comb")
    
            else:
                raise ValueError(data_train)
            
        else:
            
            if '10' in data_eval:
                from datasets.examples import get_10lamb, get_10lamb_kfold
                from datasets.default_trainingsets import xy_from_df, get_19hand, panel19withoutRightBot
    
                img_x, _ = xy_from_df(get_10lamb(), 5)
                y_val = get_10lamb_kfold().get("annot_clean_comb")

            else:
                train_data = load_data_shared(data_eval, n_per_class, seed=seed)   # TODO n_per_class is important!

                img_x, _, _, y_val = get_training_data(train_data)

        self.img_x = img_x
        # Validation data

        self.y_val = y_val

    def eval(self):

        y_pred = self.neural_net.predict(self.img_x)

        # Calculate metrics/performance

        data_i = _eval_func_single(self.y_val, y_pred, metric=metric)

        return data_i

    def safe(self, plot=False):
        
        df = pd.DataFrame(self.l)
        
        if plot:
            plot(df, 'kappa')
            plot(df, 'accuracy')
    
        # all relevant info
        l_info = [data_train]
    
        if data_eval != data_train:
            l_info.append(data_eval)
    
        l_info += [model_name, set_eval]
    
        folder_base_df = 'C:/Users/admin/OneDrive - ugentbe/data/dataframes/' if os.name == 'nt' else f'/home/lameeus/data/ghent_altar/dataframes'
    
        filepath = os.path.join(folder_base_df, f'{"_".join(l_info)}.csv')
    
        print('saving:')
        pandas_save(filepath, df, append=True)
        
        self.l = [] # Empty again!


def plot(df, metric):
    fig, ax = plt.subplots()
    for label, df_i in df.groupby(set_eval):
        df_i.sort_values('epoch').plot('epoch', metric, ax=ax, label=label)
    plt.legend()
    plt.ylabel(metric)
    plt.show()


if __name__ == '__main__':
    Main()
