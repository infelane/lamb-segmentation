import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from keras.models import load_model, model_from_json, model_from_config

from data.conversion_tools import img2array, batch2img
from datasets.default_trainingsets import get_13botleftshuang, get_19SE_shuang
from methods.basic import NeuralNet
from performance.testing import test_thresh_incremental, _get_scores, filter_non_zero
from plotting import concurrent

from main_general import get_training_data

# Stupid stuff with loading
from performance.metrics import accuracy_with0, jaccard_with0
from methods.examples import kappa_loss, weighted_categorical_crossentropy
loss = weighted_categorical_crossentropy((1, 1))


epochs_tot = 40     #40
def main():
    ### Settings
    
    mod=5
    panel_nr = 19
    
    i_start ,i_end = 1, epochs_tot
    # i_start ,i_end = 1, 2
    
    k_lst = np.arange(1, 21)
    # k_lst = [1, 2]
    
    verbose=0
    b_plot = False
    
    ###
    
    if panel_nr == 13:
        train_data = get_13botleftshuang(mod=mod)
        folder_weights = '/scratch/lameeus/data/ghent_altar/net_weight/lamb_segmentation'
    elif panel_nr == 19:
        train_data = get_19SE_shuang(mod=mod)
        folder_weights = '/scratch/lameeus/data/ghent_altar/net_weight/19_hand_SE'
    else:
        raise ValueError(panel_nr)


    x, y_tr, _, y_te = get_training_data(train_data)

    (y_tr, y_te) = map(batch2img, (y_tr, y_te))
    


    assert i_end >= i_start
    
    if b_plot:
        # plotting
        pred_lst = []
        info_lst = []
    
    lst_data = []
    lst_data_avg_pred = []
    
    for k in k_lst:
    
        model = None
        
        pred_lst = []
        
        for epoch in np.arange(i_start, i_end + 1)[::-1]:
    
            info = f'settings: k {k}; epoch {epoch}'
            print('\n\t'+info)
            
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
                
            if 1:   # Single prediction
                
                if verbose == 1:
                    print_conf(y_tr, y_pred)
                    print_conf(y_te, y_pred)
                    
                if b_plot:
                    pred_lst.append(o)
                    info_lst.append(info)
    
                test_thresh = test_thresh_incremental(y_pred, y_tr, y_te, n=5, verbose=0)
                
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
            
            if 1:   # avg prediction
    
                pred_i_average = np.mean(pred_lst, axis=0)
    
                # optimizing threshold prediction
                test_thresh = test_thresh_incremental(np.stack([1 - pred_i_average, pred_i_average], axis=-1), y_tr, y_te, n=5,
                                                      verbose=0)
                pred_thresh = np.greater_equal(pred_i_average, test_thresh)
                pred_thresh_bin = np.stack([1 - pred_thresh, pred_thresh], axis=-1)
    
                y_te_flat, y_pred_flat = filter_non_zero(y_te, pred_thresh_bin)
                y_te_argmax = np.argmax(y_te_flat, axis=-1)
                y_pred_argmax = np.argmax(y_pred_flat, axis=-1)
                acc, jacc, kappa = _get_scores(y_te_argmax, y_pred_argmax)
    
                data_i = {'k': k,
                          'epoch_start': epoch,
                          'test_thresh': test_thresh,
                          'kappa': kappa,
                          'accuracy': acc,
                          'jaccard': jacc
                          }
    
                lst_data_avg_pred.append(data_i)
            
    b = True
    if b:
        df = pd.DataFrame(lst_data)
        filename_save = f'tiunet_1pool_shaoguang{panel_nr}_imbalanced'
        filename_path = f'/scratch/lameeus/data/ghent_altar/dataframes/{filename_save}.csv'
        df.to_csv(filename_path, sep=';')
    
        df = pd.DataFrame(lst_data_avg_pred)
        filename_save = f'tiunet_1pool_shaoguang{panel_nr}_imbalanced_averaging'
        df.to_csv(f'/scratch/lameeus/data/ghent_altar/dataframes/{filename_save}.csv', sep=';')

    if b_plot:
        concurrent(pred_lst, info_lst)
    
    plt.show()
    
    return

    
def analysis():
    panel_nr = 19
    
    filename_save = f'tiunet_1pool_shaoguang{panel_nr}_imbalanced'
    df = pd.read_csv(f'/scratch/lameeus/data/ghent_altar/dataframes/{filename_save}.csv', sep=';')
    
    if 0:
        plt.figure()
        _, ax = plt.subplots()
        df.groupby(['k']).plot('epoch', 'kappa', ax=ax)
    
    if 1:
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
            
            """ Conclusion GOOD: test_thresh seems to be independent of k!"""
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
            
            # From k >= 8
            k_start = 8
            kappa_mean = df[(df['epoch']>epoch_start) & (df['k']>=k_start)]['kappa'].mean()
            
            plt.title(f'average performance (epoch>{epoch_start}) in terms of k: after 8ish it perhaps stagnate. kappa_mean = {kappa_mean}')

            b = 0
            if b:
                import tikzplotlib
                tikzplotlib.save(f"/scratch/lameeus/data/ghent_altar/tikz/perf{panel_nr}_epoch>{epoch_start}_per_k.tikz")
            
        if 0:
            """
            Correlation between test_thresh and performance?
            """
            df.corr() # Does not give interesting stuff

    plt.show()
    
    return
    
    
def analysis_average():
    panel_nr = 19
    
    filename_save = f'tiunet_1pool_shaoguang{panel_nr}_imbalanced_averaging'
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
    #         """ Conclusion GOOD: test_thresh seems to be independent of k!"""
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
        """
        Paper! for average performance

        averaged out epochs 21-...
        """
        epoch_start = 21    # Last
        
        df[df['epoch_start']==epoch_start].plot('k', ['kappa', 'accuracy', 'test_thresh'])
        
        plt.legend()

        k_start = 8
        kappa_mean = df[(df['k'] >= k_start) & (df['epoch_start'] == epoch_start)]['kappa'].mean()

        plt.title(f'Averaged out predictions. performance in terms of k. Epoch start {epoch_start}. mean(kappa), kappa>={k_start} = {kappa_mean}')

        b = 0
        if b:
            import tikzplotlib
            tikzplotlib.save(f"/scratch/lameeus/data/ghent_altar/tikz/perf{panel_nr}_averaged_prediction(k>={epoch_start})_per_k.tikz")

    if 1:
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
    b = False
    if b:
        main()
    
    b = True
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
