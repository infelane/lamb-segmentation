import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from keras.models import load_model, model_from_json, model_from_config

from data.preprocessing import img2array, batch2img
from datasets.training_examples import get_13botleftshuang
from methods.basic import NeuralNet
from performance.testing import test_incremental, _get_scores, filter_non_zero
from plotting import concurrent

from main import get_training_data

# Stupid stuff with loading
from performance.metrics import accuracy_with0, jaccard_with0
from methods.examples import kappa_loss, weighted_categorical_crossentropy
loss = weighted_categorical_crossentropy((1, 1))


def main():
    
    mod=5
    train_data = get_13botleftshuang(mod=mod)
    x, y_tr, _, y_te = get_training_data(train_data)

    (y_tr, y_te) = map(batch2img, (y_tr, y_te))

    epoch_lst = np.arange(1, 31)
    # epoch_lst = [1, 2]
    k_lst = np.arange(1, 21)
    # k_lst = [1, 2]
    
    verbose=0
    
    b_plot = False
    
    if b_plot:
        # plotting
        pred_lst = []
        info_lst = []
    
    lst_data = []
    
    for k in k_lst:
    
        model = None
        
        for epoch in epoch_lst:
    
            info = f'settings: k {k}; epoch {epoch}'
            print('\n\t'+info)
    
            folder_weights = '/scratch/lameeus/data/ghent_altar/net_weight/lamb_segmentation'
            filepath_model = os.path.join(folder_weights, f'ti_unet_k{k}_imbalanced/w_{epoch}.h5')
    
            if epoch == epoch_lst[0]:
                model = load_model(filepath_model, custom_objects={'loss': loss,
                                                                   'accuracy_with0': accuracy_with0,
                                                                   'jaccard_with0': jaccard_with0,
                                                                   'kappa_loss': kappa_loss
                                                                   })
        
            else:
                model.load_weights(filepath_model)

            n = NeuralNet(model, w_ext=10)
    
            y_pred = n.predict(x)
            o = y_pred[..., 1]
            
            def print_conf(y_true, y_pred):
                y_true = batch2img(y_true)
                y_pred = batch2img(y_pred)
                
                b_annot = np.sum(y_true, axis=-1).astype(bool)
                
                y_true_annot = y_true[b_annot, :].argmax(axis=-1)
                y_pred_annot = y_pred[b_annot, :].argmax(axis=-1)
                
                """
                T0; predicted 1, but is 0
                predicted 0, but is 1; T1
                """
                conf_mat = confusion_matrix(y_true_annot, y_pred_annot)
                print(conf_mat)
            
            if verbose == 1:
                print_conf(y_tr, y_pred)
                print_conf(y_te, y_pred)
                
            if b_plot:
                pred_lst.append(o)
                info_lst.append(info)

            test_thresh = test_incremental(y_pred, y_tr, y_te, n=5, verbose=0)
            
            pred_thresh = np.greater_equal(o, test_thresh)

            pred_thresh_bin = np.stack([1-pred_thresh, pred_thresh], axis=-1)

            y_te_flat, y_pred_flat = filter_non_zero(y_te, pred_thresh_bin)
            y_te_argmax = np.argmax(y_te_flat, axis=-1)
            y_pred_argmax = np.argmax(y_pred_flat, axis=-1)
            acc, jacc, kappa = _get_scores(y_te_argmax, y_pred_argmax)
        
            if verbose == 1:
                print_conf(y_tr, pred_thresh_bin)
                print_conf(y_te, pred_thresh_bin)

            if 0: concurrent([pred_thresh])
            
            data_i = {'k':k,
                      'epoch':epoch,
                      'test_thresh':test_thresh,
                      'kappa':kappa,
                      'accuracy':acc,
                      'jaccard':jacc
                      }
            lst_data.append(data_i)
    
    df = pd.DataFrame(lst_data)
    
    if 0:
        filename_save = 'tiunet_1pool_shaoguang13_imbalanced'
        df.to_csv(f'/scratch/lameeus/data/ghent_altar/dataframes/{filename_save}.csv', sep=';')

    if b_plot:
        concurrent(pred_lst, info_lst)
    
    plt.show()
    
    return


def main_average():
    mod = 5
    train_data = get_13botleftshuang(mod=mod)
    x, y_tr, _, y_te = get_training_data(train_data)

    (y_tr, y_te) = map(batch2img, (y_tr, y_te))

    i_start ,i_end = 1, 30
    # i_start ,i_end = 1, 2
    
    assert i_end >= i_start
    
    k_lst = np.arange(1, 21)
    # k_lst = [1, 2]

    # verbose = 0

    b_plot = False

    lst_data = []

    for k in k_lst:

        model = None

        pred_lst = []

        for epoch in np.arange(i_start, i_end+1)[::-1]:

            info = f'settings: k {k}; epoch {epoch}'
            print('\n\t' + info)

            folder_weights = '/scratch/lameeus/data/ghent_altar/net_weight/lamb_segmentation'
            filepath_model = os.path.join(folder_weights, f'ti_unet_k{k}_imbalanced/w_{epoch}.h5')

            if epoch == i_end:
                model = load_model(filepath_model, custom_objects={'loss': loss,
                                                                   'accuracy_with0': accuracy_with0,
                                                                   'jaccard_with0': jaccard_with0,
                                                                   'kappa_loss': kappa_loss
                                                                   })

            else:
                model.load_weights(filepath_model)

            n = NeuralNet(model, w_ext=10)

            y_pred = n.predict(x)
            o = y_pred[..., 1]

            pred_lst.append(o)

            pred_i_average = np.mean(pred_lst, axis=0)
            
            # optimizing threshold prediction
            test_thresh = test_incremental(np.stack([1-pred_i_average, pred_i_average], axis=-1), y_tr, y_te, n=5, verbose=0)
            pred_thresh = np.greater_equal(pred_i_average, test_thresh)
            pred_thresh_bin = np.stack([1 - pred_thresh, pred_thresh], axis=-1)

            y_te_flat, y_pred_flat = filter_non_zero(y_te, pred_thresh_bin)
            y_te_argmax = np.argmax(y_te_flat, axis=-1)
            y_pred_argmax = np.argmax(y_pred_flat, axis=-1)
            acc, jacc, kappa = _get_scores(y_te_argmax, y_pred_argmax)

            data_i = {'k': k,
                      'epoch_start': epoch,
                      'test_tresh': test_thresh,
                      'kappa': kappa,
                      'accuracy': acc,
                      'jaccard': jacc
                      }
            
            lst_data.append(data_i)
        
    b = True
    if b:
        df = pd.DataFrame(lst_data)
        
        filename_save = 'tiunet_1pool_shaoguang13_imbalanced_averaging'
        df.to_csv(f'/scratch/lameeus/data/ghent_altar/dataframes/{filename_save}.csv', sep=';')

    return

    
def analysis():
    filename_save = 'tiunet_1pool_shaoguang13_imbalanced'
    df = pd.read_csv(f'/scratch/lameeus/data/ghent_altar/dataframes/{filename_save}.csv', sep=';')
    
    if 0:
        plt.figure()
        _, ax = plt.subplots()
        df.groupby(['k']).plot('epoch', 'kappa', ax=ax)
    
    if 1:
        plt.figure()
        _, ax = plt.subplots()
        # df.groupby(['k', 'kappa']).sum()['kappa'].unstack().plot('epoch', 'kappa', ax=ax)

        for label, df_i in df.groupby('k'):
            # df_i.plot(ax=ax, label=label)
            df_i.plot('epoch', 'kappa',
                      ax=ax,
                      label=label)
        
        plt.ylim(0, None)
        plt.legend()
        
        """ Test thresholds """
        if 0:
            """ Distribution of thresholds """
            
            df.hist('test_thresh', bins=20, range=(0, 1))
            
            """ with increasing epoch, the average test_thresh goes from .5ish to .9"""
            df.groupby(['epoch']).mean()['test_thresh'].plot()
            plt.ylabel('test_thresh average')
            plt.ylim(0, 1)
            
            """ Conclusion GOOD: test_tresh seems to be independent of k!"""
            df.groupby(['k']).mean()['test_thresh'].plot()
            plt.ylabel('test_thresh average')
            plt.ylim(0, 1)
        
        """ find out point where network converged
        CONCLUSION BAD: does not look to be already converged!"""
        df.groupby(['epoch']).mean()['kappa'].plot()
        
        epoch_min = 10
        # df[df['epoch']>=epoch_min].groupby(['k']).mean()['kappa'].plot()
        #
        # df_group = df[df['epoch']>=epoch_min].groupby(['k']).expanding().mean()['kappa']

        plt.figure()
        for label, df_i in df[df['epoch']>=epoch_min].groupby(['k']):
            epoch_start = df_i['epoch']

            cummean = df_i['kappa'][::-1].expanding().mean()[::-1]
            
            plt.plot(epoch_start, cummean, label=label)
            plt.xlabel('epoch start')
            plt.ylabel('kappa')
        plt.legend()
        plt.show()
        
        if 1:
            """
            Paper! for average peformance
            
            epochs 21-30
            """
            epoch_start = 20
            df_group_k = df[df['epoch']>epoch_start].groupby('k')
            plt.figure()
            for key in ['kappa', 'accuracy', 'test_thresh']:
                plt.errorbar(df_group_k.groups.keys(), df_group_k.mean()[key],
                             yerr=df_group_k.std()[key],
                             label=key
                             )
            plt.xlabel('k')
            plt.legend()
            plt.title('performance in terms of k: after 8ish it perhaps stagnate')
        
        if 0:
            """
            Correlation between test_thresh and performance?
            """
            df.corr() # Does not give interesting stuff
            
    # df.plot('epoch', 'kappa')
    df.groupby(['k', 'kappa']).unstack().plot('epoch', 'kappa', ax=ax)
    
    return
    
    
def analysis_average():
    filename_save = 'tiunet_1pool_shaoguang13_imbalanced_averaging'
    df = pd.read_csv(f'/scratch/lameeus/data/ghent_altar/dataframes/{filename_save}.csv', sep=';')
    
    # if 0:
    #     plt.figure()
    #     _, ax = plt.subplots()
    #     df.groupby(['k']).plot('epoch', 'kappa', ax=ax)
    #
    # if 1:
    #     plt.figure()
    #     _, ax = plt.subplots()
    #     df.groupby(['k', 'kappa']).sum()['kappa'].unstack().plot('epoch_start', 'kappa', ax=ax)
    #     for label, df_i in df.groupby('k'):
    #         # df_i.plot(ax=ax, label=label)
    #         df_i.plot('epoch', 'kappa',
    #                   ax=ax,
    #                   label=label)
    #
    #     plt.ylim(0, None)
    #     plt.legend()
    #
    #     """ Test thresholds """
    #     if 0:
    #         """ Distribution of thresholds """
    #
    #         df.hist('test_thresh', bins=20, range=(0, 1))
    #
    #         """ with increasing epoch, the average test_thresh goes from .5ish to .9"""
    #         df.groupby(['epoch']).mean()['test_thresh'].plot()
    #         plt.ylabel('test_thresh average')
    #         plt.ylim(0, 1)
    #
    #         """ Conclusion GOOD: test_tresh seems to be independent of k!"""
    #         df.groupby(['k']).mean()['test_thresh'].plot()
    #         plt.ylabel('test_thresh average')
    #         plt.ylim(0, 1)
    #
    #     """ find out point where network converged
    #     CONCLUSION BAD: does not look to be already converged!"""
    #     df.groupby(['epoch']).mean()['kappa'].plot()
    #
    #     epoch_min = 10
    #     # df[df['epoch']>=epoch_min].groupby(['k']).mean()['kappa'].plot()
    #     #
    #     # df_group = df[df['epoch']>=epoch_min].groupby(['k']).expanding().mean()['kappa']
    #
    #     plt.figure()
    #     for label, df_i in df[df['epoch'] >= epoch_min].groupby(['k']):
    #         epoch_start = df_i['epoch']
    #
    #         cummean = df_i['kappa'][::-1].expanding().mean()[::-1]
    #
    #         plt.plot(epoch_start, cummean, label=label)
    #         plt.xlabel('epoch start')
    #         plt.ylabel('kappa')
    #     plt.legend()
    #     plt.show()
    
    if 1:
        # df.groupby('epoch_start').mean().plot('kappa')
        
        """
        Paper! By averaging out prediction this is the performance (single value in the end for table)
        """
    
        df_group = df.groupby('epoch_start')
        plt.figure()
        for key in ['kappa', 'accuracy', 'test_thresh']:
            plt.errorbar(df_group.groups.keys(), df_group.mean()[key],
                         yerr=df_group.std()[key],
                         label=key
                         )
        plt.xlabel('epoch start of averaging out prediction')
        plt.legend()
        plt.title('performance grouped per k of averaging out predictions (some sort of majority voting)')
        plt.show()
        
    if 0:
        """
        Paper! for average performance

        epochs 21-30
        """
        epoch_start = 1
        df_group_k = df[df['epoch_start'] >= epoch_start].groupby('k')
        plt.figure()
        for key in ['kappa', 'accuracy', 'test_thresh']:
            plt.errorbar(df_group_k.groups.keys(), df_group_k.mean()[key],
                         yerr=df_group_k.std()[key],
                         label=key
                         )
        plt.xlabel('k')
        plt.legend()
        plt.title('performance in terms of k')

    #     if 0:
    #         """
    #         Correlation between test_thresh and performance?
    #         """
    #         df.corr()  # Does not give interesting stuff
    #
    # # df.plot('epoch', 'kappa')
    # df.groupby(['k', 'kappa']).unstack().plot('epoch', 'kappa', ax=ax)
    
    return


if __name__ == '__main__':
    b = True
    if b:
        main()
    main_average()
    
    b = False
    if b:
        analysis()
    analysis_average()


import tensorflow as tf
import keras.backend as K
def weighted_categorical_crossentropy(weights):
    """ weighted_categorical_crossentropy

        Args:
            * weights<ktensor|nparray|list>: crossentropy weights
        Returns:
            * weighted categorical crossentropy function
    """

    if not isinstance(weights, tf.Variable):
        weights = K.variable(weights)

    def loss(target, output, from_logits=False):
        if not from_logits:
            output /= tf.reduce_sum(output,
                                    len(output.get_shape()) - 1,
                                    True)
            _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            weighted_losses = target * tf.log(output) * weights
            return - tf.reduce_sum(weighted_losses, len(output.get_shape()) - 1)
        else:
            raise ValueError('WeightedCategoricalCrossentropy: not valid with logits')

    return loss
