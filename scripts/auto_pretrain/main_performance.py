import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.models import load_model, Model

from data.conversion_tools import img2array, batch2img
from data.preprocessing import rescale0to1
from datasets.default_trainingsets import get_10lamb_all, get_10lamb_6patches
from main_general import get_training_data
from methods.examples import get_neural_net_ae, neuralNet0
from methods.basic import NeuralNet
from neuralNetwork.optimization import find_learning_rate
from performance.testing import optimal_test_thresh_equal_distribution, test_thresh_incremental, _get_scores, filter_non_zero
from plotting import concurrent
from preprocessing.image import get_class_weights, get_class_imbalance, get_flow

# Loading the model
from methods.examples import kappa_loss, weighted_categorical_crossentropy
from performance.metrics import accuracy_with0, jaccard_with0
loss = weighted_categorical_crossentropy((1, 1))


def main():
    """

    :return:
    """
    
    ### Settings
    
    k_range = np.arange(2, 30 + 1).astype(int)
    # k_range = [10,11]

    fold_range = np.arange(6).astype(int)
    # fold_range = [0, 1]

    epoch_range = np.arange(1, 40 + 1).astype(int)
    # epoch_range = [39, 40]

    filename_single = f'tiunet_10lamb_kfold_single'
    filename_avg_pred =f'tiunet_10lamb_kfold_avgpred'

    if os.name == 'nt':     # windows laptop
        folder_weights = 'C:/Users/Laurens_laptop_w/data'
        folder_save = 'C:/Users/Laurens_laptop_w/data/ghent_altar/dataframes'
    else:
        folder_weights = '/scratch/lameeus/data/ghent_altar/net_weight'
        folder_save = '/home/lameeus/data/ghent_altar/dataframes'

    ### Init
    epoch_range_desc = np.sort(epoch_range)[::-1]

    k_fold_train_data = get_10lamb_6patches(5)  # 5 is the number of modalities
    train_data_all = k_fold_train_data.get_train_data_all()
    img_x = train_data_all.get_x_train()
    img_x = rescale0to1(img_x)
    img_y_all = train_data_all.get_y_train()
    
    ###
    lst_data_single = []
    lst_data_avg_pred = []
    
    for k in k_range:
        for i_fold in fold_range:
            
            ### Reinit to make sure
            model = None
            list_y_pred = []

            train_data_i = k_fold_train_data.k_split_i(i_fold)
            img_y_tr = train_data_i.get_y_train()
            img_y_te = train_data_i.get_y_test()
            
            for epoch in epoch_range_desc:
    
                """
                Load model
                """
                filepath_model = os.path.join(folder_weights, f'10lamb_kfold/ti_unet_k{k}_kfold{i_fold}/w_{epoch}.h5')

                if epoch == epoch_range_desc[0]:
                    model = load_model(filepath_model, custom_objects={'loss': loss,
                                                                       'accuracy_with0': accuracy_with0,
                                                                       'jaccard_with0': jaccard_with0,
                                                                       'kappa_loss': kappa_loss
                                                                       })

                    assert len(list_y_pred) == 0
                    
                else:
                    model.load_weights(filepath_model)

                """
                Inference
                """
        
                n = NeuralNet(model, w_ext=10)
                y_pred = n.predict(img_x)
                
                """
                Average out predictions
                """
                list_y_pred.append(y_pred)

                y_avg_pred = np.mean(list_y_pred, axis=0)
                
                """
                thresh based on GT
                """

                thresh_single = optimal_test_thresh_equal_distribution(img_y_all, y_pred)
                thresh_avg_pred = optimal_test_thresh_equal_distribution(img_y_all, y_avg_pred)
                
                """
                Get scores
                """
                
                data_single_i = {'k': k,
                                 'i_fold': i_fold,
                                 'epoch': epoch}
                data_avg_pred_i = {'k': k,
                                   'i_fold': i_fold,
                                   'epoch_start': epoch}
                
                data_single_i.update(foo_performance(img_y_te, y_pred, thresh_single))
                data_avg_pred_i.update(foo_performance(img_y_te, y_avg_pred, thresh_avg_pred))
                
                if 1:
                    print('single', data_single_i)
                    print('avg pred', data_avg_pred_i)

                lst_data_single.append(data_single_i)
                lst_data_avg_pred.append(data_avg_pred_i)
        
            """
            Save data
            """
            
            df_single = pd.DataFrame(lst_data_single)
            df_avg_pred = pd.DataFrame(lst_data_avg_pred)
            
            path_single = os.path.join(folder_save, filename_single + '.csv')
            path_avg_pred = os.path.join(folder_save, filename_avg_pred + '.csv')
            if os.path.exists(path_single):
                df_single.to_csv(path_single, mode='a', header=False, index=False)
            else:
                df_single.to_csv(path_single, index=False)

            if os.path.exists(path_avg_pred):
                df_avg_pred.to_csv(path_avg_pred, mode='a', header=False, index=False)
            else:
                df_avg_pred.to_csv(path_avg_pred, index=False)
    
    return

    
def foo_performance(y_true, y_pred, thresh):
    """
    
    :param y:
    :param y_gt:
    :return:
    """
    
    assert y_true.shape[-1] == y_pred.shape[-1] == 2
    
    """
    to indexing
    """

    # Only work where we have annotations from
    y_true_flat, y_pred_flat = filter_non_zero(y_true, y_pred)
    
    y_true_argmax = y_true_flat[..., 1]
    y_pred_argmax = np.greater_equal(y_pred_flat[..., 1], thresh)
    
    """
    Scores
    """

    acc, jacc, kappa = _get_scores(y_true_argmax, y_pred_argmax)
    
    return {'thresh_equal_distr': thresh,
            'accuracy': acc,
            'jaccard': jacc,
            'kappa': kappa}
    

if __name__ == '__main__':
    main()
