import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.datatools import pandas_save
from data.preprocessing import rescale0to1
from datasets.default_trainingsets import get_10lamb_all, get_10lamb_6patches
from figures_paper.overlay import semi_transparant
from methods.basic import NeuralNet
from performance.testing import optimal_test_thresh_equal_distribution, test_thresh_incremental, _get_scores, filter_non_zero
from plotting import concurrent

# ### Loading the model
# from methods.examples import kappa_loss, weighted_categorical_crossentropy
# from performance.metrics import accuracy_with0, jaccard_with0
# loss = weighted_categorical_crossentropy((1, 1))

from scripts.auto_pretrain.main_performance import load_model_quick, foo_performance


def main():
    
    b_encoder_fixed = False
    info_enc_fixed = '_enc_fixed'
    
    folder_weights = '/scratch/lameeus/data/ghent_altar/net_weight/10lamb_kfold_pretrained'
    folder_save = '/home/lameeus/data/ghent_altar/dataframes'
    filename_single = f'pretrained_unet_10lamb_kfold_single'
    filename_avg_pred = f'pretrained_unet_10lamb_kfold_avgpred'
    folder_weights += info_enc_fixed if b_encoder_fixed else ''
    filename_single += info_enc_fixed if b_encoder_fixed else ''
    filename_avg_pred += info_enc_fixed if b_encoder_fixed else ''

    fold_range = range(6)
    # fold_range = [0, 1]
    
    k = 10
    epoch_range = range(1, 40 + 1)
    
    w_ext_in = 28

    k_fold_train_data = get_10lamb_6patches(5)  # 5 is the number of modalities
    train_data_all = k_fold_train_data.get_train_data_all()
    img_x = train_data_all.get_x_train()
    img_x = rescale0to1(img_x)
    img_clean = img_x[..., :3]
    img_y_all = train_data_all.get_y_train()
    
    b_plot = False


    for i_fold in fold_range:
    
        print(i_fold)

        img_y_te = k_fold_train_data.k_split_i(i_fold).get_y_test()

        # Init for range epochs
        lst_data_single = []
        lst_data_avg_pred = []
        list_y_pred = []
        model = None
        
        for epoch in np.sort(epoch_range)[::-1]:
            

            
            filepath_model = os.path.join(folder_weights, f'unet_enc_k{k}_ifold{i_fold}/w_{epoch}.h5')

            model = load_model_quick(filepath_model, model=model)
            n = NeuralNet(model, w_ext=w_ext_in)
            y_pred = n.predict(img_x)

            """
            Average out predictions
            """
            list_y_pred.append(y_pred)
            y_avg_pred = np.mean(list_y_pred, axis=0)

            thresh_single = optimal_test_thresh_equal_distribution(img_y_all, y_pred)
            thresh_avg_pred = optimal_test_thresh_equal_distribution(img_y_all, y_avg_pred)

            y_pred_bin = np.greater_equal(y_pred[..., 1], thresh_single)

            dict_perf = foo_performance(img_y_te, y_pred, thresh_single)
            print(dict_perf)
            
            if b_plot:
                concurrent([y_pred_bin, img_clean,
                            semi_transparant(img_clean, y_pred_bin),
                            semi_transparant(img_clean, img_y_te[..., 1].astype(bool))])

            data_single_i = {'k': k,
                             'i_fold': i_fold,
                             'epoch': epoch}
            data_avg_pred_i = {'k': k,
                               'i_fold': i_fold,
                               'epoch_start': epoch,
                               'epoch_end': max(epoch_range)}
            
            data_single_i.update(dict_perf)
            data_avg_pred_i.update(foo_performance(img_y_te, y_avg_pred, thresh_avg_pred))

            lst_data_single.append(data_single_i)
            lst_data_avg_pred.append(data_avg_pred_i)

        df_single = pd.DataFrame(lst_data_single)
        df_avg_pred = pd.DataFrame(lst_data_avg_pred)

        path_single = os.path.join(folder_save, filename_single + '.csv')
        path_avg_pred = os.path.join(folder_save, filename_avg_pred + '.csv')

        pandas_save(path_single, df_single, append=True)
        pandas_save(path_avg_pred, df_avg_pred, append=True)
        
    return


if __name__ == '__main__':
    main()

    