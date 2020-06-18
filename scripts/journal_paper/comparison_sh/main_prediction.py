import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts import standard_keras2

from figures_paper.comparison_shuang import get_crop
from methods.basic import NeuralNet
from plotting import concurrent
from scripts.scripts_performance.main_performance import load_model_quick
from scripts.main_simple import _eval_func_single

data = '19botright'     # 1319botright, 19botrightcrack 19botrightcrack13 '13botright', '19botright'

# Evaluation data:
from datasets.examples import get_10lamb, get_10lamb_kfold
from datasets.default_trainingsets import xy_from_df
from data.conversion_tools import annotations2y


def influence_n_per_class():
    fig, ax = plt.subplots()

    metric = 'accuracy' # kappa

    basefolder = 'C:/Users/admin/OneDrive - ugentbe/data/dataframes'

    for model_name in ['ti-unet', 'unet']:
        df = pd.read_csv(os.path.join(basefolder, f'{data}_{model_name}_n_per_class.csv'), delimiter=';')

        # for label, df_i in df.groupby(set_eval):

        idx = df.groupby('n_per_class')[metric].idxmax()

        df.loc[idx].sort_values('n_per_class').plot('n_per_class', metric, ax=ax, label=model_name)

        # df.sort_values('n_per_class').plot('n_per_class', 'kappa', ax=ax, label=model_name)

    plt.legend()
    plt.ylabel(metric)

    plt.xscale('log')

    if 0:
        import tikzplotlib
        tikzplotlib.save(os.path.join(basefolder, "test.tikz"))

    plt.show()



    return


def transfer_learning_init():

    img_x, img_y_val = data_lamb()

    d = 2
    k = 10
    model_name = 'ti-unet'

    train_data = '1319'

    w_ext = 10 if d == 1 else 26

    y_pred_lst = []
    n = []
    for epoch in range(10, 101, 10):

        path = f'C:/Users/admin/Data/ghent_altar/net_weight/{train_data}/{model_name}_d{d}_k{k}/w_{epoch}.h5'

        try:
            model = load_model_quick(path)
        except Exception as e:
            print(e)
            continue

        neural_net = NeuralNet(model, w_ext=w_ext, norm_x=True)

        y_pred = neural_net.predict(img_x)

        if 0:
            data_i = _eval_func_single(img_y_val, y_pred, metric='kappa')
            print(data_i)
            data_i = _eval_func_single(img_y_val, y_pred, metric='jaccard')
            print(data_i)

        y_pred_lst.append(y_pred)
        n.append(epoch)

    concurrent([a[..., 1] for a in y_pred_lst], n)
    plt.show()

    return 1


def pred_epochs():
    img_x, img_y_val = data_lamb()

    d = 2
    k = 10
    model_name = 'ti-unet'

    train_data = '1319_10nat'

    w_ext = 10 if d == 1 else 26

    y_pred_lst = []
    n = []
    for epoch in range(10, 101, 10):
        print(epoch)

        epoch_start = 50
        epoch_corr = epoch + epoch_start if train_data[:5] == '1319_' else epoch
        path = f'C:/Users/admin/Data/ghent_altar/net_weight/{train_data}/{model_name}_d{d}_k{k}/w_{epoch_corr}.h5'

        try:
            model = load_model_quick(path)
        except Exception as e:
            print(e)
            continue

        neural_net = NeuralNet(model, w_ext=w_ext, norm_x=True)

        y_pred = neural_net.predict(img_x)

        if 0:
            data_i = _eval_func_single(img_y_val, y_pred, metric='kappa')
            print(data_i)
            data_i = _eval_func_single(img_y_val, y_pred, metric='jaccard')
            print(data_i)

        y_pred_lst.append(y_pred)
        n.append(epoch)

    concurrent([a[..., 1] for a in y_pred_lst], n)
    plt.show()

    return 1


def transfer_learning(epoch=25,     # Could check a few
                      b_plot=False
                      ):

    d = 2  # 1, 2

    img_x, img_y_val = data_lamb()

    k = 10

    model_name = 'ti-unet'

    w_ext = 10 if d == 1 else 26

    # train_data:
    y_pred_lst = []
    n = ['clean']

    # train_data_lst = ['1319_10', '10', '1319', '1319_101319']
    train_data_lst = ['10nat', '1319_10nat', '1319_10nat1319', '1319']

    data_i_lst = {}

    for train_data in train_data_lst:
        print(train_data)

        epoch_start = 50
        epoch_corr = epoch + epoch_start if train_data[:5] == '1319_' else epoch
        if train_data == '1319':
            epoch_corr = 50
        path = f'C:/Users/admin/Data/ghent_altar/net_weight/{train_data}/{model_name}_d{d}_k{k}/w_{epoch_corr}.h5'

        try:
            model = load_model_quick(path)
        except Exception as e:
            print(e)
            continue

        neural_net = NeuralNet(model, w_ext=w_ext, norm_x=True)

        y_pred = neural_net.predict(img_x)

        # baseline
        data_i = _eval_func_single(img_y_val, y_pred, metric='kappa')
        print(data_i)

        if 0:
            """
            Checking which 
            
            baseline ~ .22
            i = 0: .268, Remove huge improvement  ( a lot of "green" background annotated as paint loss)
            i = 1: .228 Keep!
            i = 2: .179 keep! Drop (keep!!
            i = 3: .159 keep! Even more important
            i = 4: .252 Remove (huge problem right top)
            i = 5: .233 Keep, quit relevant
            """

            from datasets.default_trainingsets import get_10lamb_6patches
            kFoldTrainData = get_10lamb_6patches(5)

            _eval_func_single(kFoldTrainData.k_split_i(0).get_y_train(), y_pred, metric='kappa')    # Check what is influence without!

        data_i_lst[train_data] = data_i

        data_i = _eval_func_single(img_y_val, y_pred, metric='jaccard')
        print(data_i)

        y_pred_lst.append(y_pred)
        n.append(train_data)

    # plt.imshow(neural_net.predict(img_x[::2,::2,:])[..., 1])

    if b_plot:
        concurrent([img_x[..., :3]] + [a[..., 1] for a in y_pred_lst], n)

    if 0:
        from figures_paper.overlay import semi_transparant
        from data.datatools import imread, imsave

        t = [data_i_lst[n_i]['thresh'] for n_i in train_data_lst]
        p = []
        for i, train_data in enumerate(train_data_lst):
            b = np.greater_equal(y_pred_lst[i][..., 1], t[i])

            k = semi_transparant(img_x[..., :3], b, 0)
            p.append(k)

            imsave(os.path.join("C:/Users/admin/OneDrive - ugentbe/data/images_paper", train_data + '.png'), k)

        concurrent(p)

    return data_i_lst


def data_lamb():
    img_x, _ = xy_from_df(get_10lamb(), 5)

    if 0:
        img_y_val = annotations2y(get_10lamb_kfold().get("annot_clean_comb"))
    else:
        # "Improved" evaluation set

        from datasets.default_trainingsets import get_10lamb_6patches
        kFoldTrainData = get_10lamb_6patches(5)

        img_y_val_lst = []

        for i in [1, 2, 3, 5]:
            y_i = kFoldTrainData.k_split_i(i).get_y_test()

            img_y_val_lst.append(y_i)

        img_y_val = np.sum(img_y_val_lst, axis=0)

    return img_x, img_y_val


def eval_3outputs():
    
    folder_base = 'C:/Users/admin/Data/ghent_altar/' if os.name == 'nt' else '/scratch/lameeus/data/ghent_altar/'

    assert data == '19botrightcrack3'
    
    k = 9
    epoch = 25
    model_name = 'ti-unet'

    folder_base = 'C:/Users/admin/Data/ghent_altar/' if os.name == 'nt' else '/scratch/lameeus/data/ghent_altar/'
    path = os.path.join(folder_base, f'net_weight/{data}/{model_name}_d1_k{k}_n80/w_{epoch}.h5')

    model = load_model_quick(path)
    neural_net = NeuralNet(model, w_ext=10, norm_x=True)
    
    from scripts.journal_paper.comparison_sh.shared import load_data
    a = load_data("19botright", 80)
    img_x, y_eval = a.get_x_train(), a.get_y_test()

    y_pred = neural_net.predict(img_x)
    
    assert y_pred.shape[-1] == 3
    y_pred2 = np.stack([1 - y_pred[..., 1], y_pred[..., 1]], axis=-1)

    data_i = _eval_func_single(y_eval, y_pred2)
    
    print(data_i)
    
    return


class Evaluater(object):
    def __init__(self, y_true, y_pred, metric="kappa"):
        """
        With automated threshold that optimizes metric
        :param y_true:
        :param y_pred:
        """

        assert np.all(np.equal(y_true.astype(bool).astype(int), y_true))

        assert y_true.shape[-1] == y_pred.shape[-1] == 2

        self.y_true = y_true
        self.y_pred = y_pred

        self.metric = metric

    def get_prod_acc(self):

        try:
            self.prod_acc
        except AttributeError:
            self._auto_eval()
            self._auto_cm()

        return self.prod_acc

    def get_user_acc(self):

        try:
            self.user_acc
        except AttributeError:
            self._auto_eval()
            self._auto_cm()

        return self.user_acc

    def get_oa(self):
        try:
            self.acc
        except AttributeError:
            self._auto_eval()

        return self.acc

    def get_kappa(self):

        try:
            self.kappa
        except AttributeError:
            self._auto_eval()

        return self.kappa

    def get_jaccard(self):

        try:
            self.jaccard
        except AttributeError:
            self._auto_eval()

        return self.jaccard

    def _auto_eval(self):
        data_i = _eval_func_single(self.y_true, self.y_pred, metric=self.metric)

        self.t = data_i["thresh"]
        self.kappa = data_i["kappa"]
        self.acc = data_i["accuracy"]
        self.jaccard = data_i["jaccard"]

    def _auto_cm(self):
        # TODO with threshold calculate user acc
        from sklearn.metrics import confusion_matrix

        from performance.testing import filter_non_zero, get_y_pred_thresh

        y_true_filter, y_pred_filter = filter_non_zero(self.y_true, self.y_pred)
        y_pred_filter_thresh = get_y_pred_thresh(y_pred_filter, self.t)

        y_true_arg = np.argmax(y_true_filter, axis=-1)
        y_pred_arg = np.argmax(y_pred_filter_thresh, axis=-1)

        cm = confusion_matrix(y_true_arg, y_pred_arg)

        diag = np.diag(cm)

        # producer accuracy: TP/n_GT
        prod = np.sum(cm, axis=0)  # Sum along first axis (sum of each column) Producer

        self.prod_acc = diag/prod

        # User accuracy: TP/n_classified
        user = np.sum(cm, axis=1)  # Sum along second axis (sum of each row)   User

        self.user_acc = diag/user
    def summary(self):

        print("oa =", self.get_oa())
        print("prod acc =", self.get_prod_acc())
        print("user acc =", self.get_user_acc())
        print("kappa =", self.get_kappa())
        print("jaccard =", self.get_jaccard())

if __name__ == '__main__':
    """
    Decide which model to take
    """

    if 0:
        influence_n_per_class()     # Only check this for the moment

    elif 0: # 1!
        
        # What is a good init?
        if 0:
            transfer_learning_init()

        if 0:
            pred_epochs()

        if 0:   # evaluate for the different epochs:
            data_i_all = {}
            for e in range(1, 101, 1):
                print('epoch', e)
                data_i_all[e] = transfer_learning(e)

            print(data_i_all)
            # TODO "process"

            keys_e = list(data_i_all.keys())
            keys_e.sort()
            keys_train_data = list(data_i_all[keys_e[0]].keys())

            plt.figure()
            metric = 'kappa'
            for train_data_i in keys_train_data:

                y = [data_i_all[e][train_data_i][metric] for e in keys_e]

                plt.plot(keys_e, y, label=train_data_i)

            plt.xlabel('epoch')
            plt.ylabel(metric)
            plt.legend()
            if 0:
                import tikzplotlib
                tikzplotlib.save("C:/Users/admin/OneDrive - ugentbe/data/dataframes/transfer_learning.tikz")
            plt.show()

            l_df = []
            for key in data_i_all:
                for key2 in data_i_all[key]:
                    data_i = data_i_all[key][key2]

                    data_i['epoch'] = key
                    data_i['dataset'] = key2
                    l_df.append(data_i)

            df = pd.DataFrame(l_df)

            if 0:
                from data.datatools import pandas_save
                filepath = f'C:/Users/admin/OneDrive - ugentbe/data/dataframes/transfer_learning_lamb4.csv'
                print('saving:')
                pandas_save(filepath, df, append=True)

        transfer_learning(b_plot=True)

    elif 0:
        eval_3outputs()

    folder_base_df = 'C:/Users/admin/OneDrive - ugentbe/data/dataframes/' if os.name == 'nt' else f'/home/lameeus/data/ghent_altar/dataframes'

    df = pd.read_csv(os.path.join(folder_base_df, f'{data}_{"ti-unet"}_n_per_class.csv'), delimiter=';')

    i_max = df['kappa'].idxmax()

    k, epoch = map(int, df.iloc[i_max][['k', 'epoch']])

    model_name = 'ti-unet'

    folder_base = 'C:/Users/admin/Data/ghent_altar/' if os.name == 'nt' else '/scratch/lameeus/data/ghent_altar/'

    path = os.path.join(folder_base, f'net_weight/{data}/{model_name}_d1_k{k}_n80/w_{epoch}.h5')

    model = load_model_quick(path)
    neural_net = NeuralNet(model, w_ext=10, norm_x=True)

    # Image
    
    from scripts.journal_paper.comparison_sh.shared import load_data

    if data == '1319botright':
        a = load_data("19botright", n_per_class=80)
    else:
        a = load_data(data, n_per_class=80)
    img_x, img_y = a.get_x_train(), a.get_y_test()

    y_pred = neural_net.predict(img_x)

    if 1:
        Evaluater(img_y, y_pred).summary()
        print("For the tables!")

    def average_out_pred(r = 2):
        model_name = 'ti-unet'
        
        path = os.path.join(folder_base, f'net_weight/{data}/{model_name}_d1_k{k}_n80/w_{1}.h5')
        model_i = load_model_quick(path)

        neural_net_i = NeuralNet(model_i, w_ext=10, norm_x=True)
        
        y_pred_lst = []
        r = 2
        for epoch_i in range(epoch-r, epoch+r+1): # epochs
            
            neural_net_i.load(path.rsplit('/',1)[0], epoch_i) # Load
            
            try:
                y_pred_i = neural_net_i.predict(img_x)
            except Exception as e:
                print(e)
                continue

            y_pred_lst.append(y_pred_i[..., 1])
        y_pred_avg = np.mean(y_pred_lst, axis=0)
        
        return y_pred_avg
        
    if 1:
        # Average out prediction

        y_pred_avg = average_out_pred()
        
        if 0:
            concurrent([y_pred_avg])

        y_pred_avg2 = np.stack([1-y_pred_avg, y_pred_avg], axis=-1)

        data_i = _eval_func_single(get_crop(img_y), get_crop(y_pred_avg2))
        print(data_i)

    from performance.testing import get_y_pred_thresh

    thresh = df.iloc[i_max]['thresh']

    y_thresh = get_y_pred_thresh(y_pred, thresh=thresh)

    concurrent([img_x[..., :3], y_pred[..., 0], y_thresh[..., 0]])

    im_pred = get_crop(y_thresh)
    b_pred = im_pred[..., 1]

    # For U-Net
    model_name = 'unet'
    df_unet = pd.read_csv(f'C:/Users/admin/OneDrive - ugentbe/data/dataframes/{data}_{model_name}.csv', delimiter=';')
    i_max_unet = df_unet['kappa'].idxmax()
    k_unet, epoch_unet = map(int, df_unet.iloc[i_max_unet][['k', 'epoch']])
    thresh_unet = df_unet.iloc[i_max_unet]['thresh']
    path = f'C:/Users/admin/Data/ghent_altar/net_weight/{data}/{model_name}_d1_k{k_unet}_n80/w_{epoch_unet}.h5'
    model_unet = load_model_quick(path)
    neural_net_unet = NeuralNet(model_unet, w_ext=10, norm_x=True)
    y_pred_unet = neural_net_unet.predict(img_x)
    y_thresh_unet = get_y_pred_thresh(y_pred_unet, thresh=thresh_unet)
    im_pred_unet = get_crop(y_thresh_unet)
    b_pred_unet = im_pred_unet[..., 1]

    # TODO double check performance!

    import os
    folder = 'C:/Users/admin/OneDrive - ugentbe/Documents/2019journal paper/2020_04_06/images/results_compare'
    from data.datatools import imread, imsave

    # Annot

    if '13' in data:
        data_nr = 13
    elif '19' in data:
        data_nr = 19
    else:
        raise ValueError(data)

    im_clean = imread(os.path.join(folder, f'{data_nr}_3_clean.jpg'))

    im_annot = imread(os.path.join(folder,  f'{data_nr}_3_annot.png'))

    im_sh = imread(os.path.join(folder,  f'{data_nr}_3_src.png'))

    # TODO get binary

    from data.conversion_tools import detect_colour

    b_annot = detect_colour(im_annot, 'cyan', thresh=.9)

    b_sh = detect_colour(im_sh, 'cyan', thresh=.9)

    # Performance measures:

    concurrent([b_sh, b_pred, b_pred_unet])
    for _, b in enumerate([b_sh, b_pred, b_pred_unet]):
        data_i = _eval_func_single(np.stack([1-b_annot, b_annot], axis=-1), np.stack([1-b, b], axis=-1))
        print(data_i)

    # Show before saving all

    from figures_paper.overlay import semi_transparant

    l = [im_clean]

    for i, b in enumerate([b_annot, b_sh, b_pred, b_pred_unet]):

        if i == 0:
            # Probably not really interesting to change the colour
            l.append(semi_transparant(im_clean, b.astype(bool), color1='cyan', color2='grey', transparency=0, transparancy2=.5))
        else:
            l.append(semi_transparant(im_clean, b.astype(bool), color2='grey', transparency=0, transparancy2=.5))

    concurrent(l)

    folder_out = 'C:/Users/admin/OneDrive - ugentbe/data/images_paper'
    imsave(os.path.join(folder_out, f'{data}_clean.png'), im_clean)
    imsave(os.path.join(folder_out, f'{data}_annot.png'), l[1])
    imsave(os.path.join(folder_out, f'{data}_src.png'), l[2])
    imsave(os.path.join(folder_out, f'{data}_tiunet.png'), l[3])
    imsave(os.path.join(folder_out, f'{data}_unet.png'), l[4])
