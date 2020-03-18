import os

import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model, Model

# from data.conversion_tools import img2array, batch2img, img2batch
from data.preprocessing import rescale0to1
from datasets.default_trainingsets import get_10lamb_old, get_10lamb_6patches, get_13
from datasets.script_combine_10lamb_6annot import get_borders1, get_borders2, get_borders3, get_borders4, get_borders5, get_borders6
from main_general import get_training_data
# from methods.examples import get_neural_net_ae, neuralNet0
from methods.basic import NeuralNet
# from neuralNetwork.optimization import find_learning_rate
from performance.testing import optimal_test_thresh_equal_distribution, test_thresh_incremental
from plotting import concurrent
# from preprocessing.image import get_class_weights, get_class_imbalance, get_flow
from figures_paper.overlay import semi_transparant
from scripts.scripts_performance.main_performance import foo_performance

from neuralNetwork.architectures import ti_unet
from data.modalities import get_mod_n_in
from methods.examples import compile_segm

from data.datatools import imsave


class Main(object):
    w_patch = 500
    fixed_enc = 0
    
    set = 13    # 10
    
    def __init__(self):
        self.set_data()
        
        if 1:
            y_lst = []
            for i_fold in range(6):
                for k in [12, 13]:
                    for epoch in [99, 100]:
                        y = self.predict(i_fold=0, k=k, epoch=epoch)

                        y_lst.append(y)
                        
            y_avg = np.mean(y_lst, axis=0)
            y_avg1 = y_avg[..., 1]
            plt.imshow(y_avg1)
            if 0: imsave(f'/home/lameeus/data/ghent_altar/output/hierarchy/13_small/pred_transfer_kfoldenc{self.fixed_enc}_avg.png',
                         y_avg1)
            
        if 0:
            self.predict_average()
        if 0:
            self.predict_compare()

        if 0:
            # regular TI-UNet
            self.predict_compare_regular()
    
    def set_data(self):
        
        if self.set == 10:
            train_data = get_10lamb_old(5)
            img_x, _, _, _ = get_training_data(train_data)
        elif self.set == 13:
            img_x, _, _, _ = get_training_data(get_13(5))
            
        # Normalise the input!
        img_x = rescale0to1(img_x)
        self.img_x = img_x
        
        self.k_fold_train_data = get_10lamb_6patches(5)
        
    def get_n_model(self,
                    i_fold=None,
                    k=None,
                    epoch = None
                    ):
        
        n_in = 9
        
        model_tiunet = ti_unet(n_in, filters=k,
                               w=self.w_patch,
                               ext_in=10 // 2,
                               batch_norm=True)

        n = NeuralNet(model_tiunet, w_ext=10)

        info_batchnorm = '_batchnorm'
        info_fixed = '_encfixed' if self.fixed_enc == 1 else '_prefixed' if self.fixed_enc == 2 else ''
        info_model = 'tiunet'
        info = f'10lamb_kfold_pretrained{info_fixed}{info_batchnorm}/{info_model}_d{1}_k{k}_ifold{i_fold}'
        folder = os.path.join('/scratch/lameeus/data/ghent_altar/net_weight', info)
        n.load(folder=folder, epoch=epoch)
        
        return n

    def get_n_model_regular(self,
                    i_fold=None,
                    k=None,
                    epoch = None
                    ):
        
        n_in = 9

        model_tiunet = ti_unet(n_in, filters=k,
                               w=self.w_patch,
                               ext_in=10 // 2,
                               batch_norm=True,
                               wrong_batch_norm=True)

        """ TODO wrong tiunet """

        n = NeuralNet(model_tiunet, w_ext=10)

        info = f'10lamb_kfold/ti_unet_k{k}_kfold{i_fold}'
        folder = os.path.join('/scratch/lameeus/data/ghent_altar/net_weight', info)
        n.load(folder=folder, epoch=epoch)

        return n
        
    def predict(self, i_fold=None, k=None, epoch=None):
        
        n = self.get_n_model(i_fold=i_fold, k=k, epoch=epoch)
        
        y = n.predict(self.img_x)
        
        if 0:
            concurrent([y[..., 1], self.img_x[..., :3]])
        
        return y
    
    def predict_regular(self, i_fold=None, k=None, epoch=None):
        
        n = self.get_n_model_regular(i_fold=i_fold, k=k, epoch=epoch)

        y = n.predict(self.img_x)

        if 0:
            concurrent([y[..., 1], self.img_x[..., :3]])

        return y
    
    def predict_average(self):
        
        y_lst = []
        for i_fold in range(6):
            for k in range(18, 19+1):
                for epoch in range(99, 100+1):
            
                    y = self.predict(i_fold=i_fold, k=k, epoch=epoch)
                    y_lst.append(y)
        y_avg = np.mean(y_lst, axis=0)
        
        y_avg_bin = self.get_bin(y_avg)

        img_clean = self.img_x[..., :3]

        y_avg_bin_overlay = semi_transparant(img_clean, y_avg_bin)
        
        concurrent([y_avg[..., 1],
                    y_avg_bin,
                    y_avg_bin_overlay,
                    img_clean])
        
        if 0:
            path_save = f'/scratch/lameeus/data/ghent_altar/output/hierarchy/10_lamb/' \
                        f'paintloss_tiunet_enc{self.fixed_enc}.png'
            imsave(path_save, y_avg_bin)
        if 1:
            path_save = f'/scratch/lameeus/data/ghent_altar/output/hierarchy/10_lamb/' \
                        f'paintloss_overlay_tiunet_enc{self.fixed_enc}.png'
            imsave(path_save, y_avg_bin_overlay)
        
        return 1

    def predict_compare(self):
    
        img_y_all = self.k_fold_train_data.get_train_data_all().get_y_train()
        lst_get = [get_borders1, get_borders2, get_borders3, get_borders4, get_borders5, get_borders6]
        img_clean = self.img_x[..., :3]

        for i_fold in range(6):
            for i_fixed_enc in range(3):
                self.fixed_enc = i_fixed_enc
                
                y_lst = []
                # Average prediction
                for k in [17, 18, 19]:
                    for epoch in [96, 97, 98, 99, 100]:
    
                        y = self.predict(i_fold=i_fold, k=k, epoch=epoch)
                        y_lst.append(y)

                y_avg = np.mean(y_lst, axis=0)
                y_avg_bin = self.get_bin(y_avg)
                
                # Performance
                img_y_te = self.k_fold_train_data.k_split_i(i_fold).get_y_test()

                thresh = optimal_test_thresh_equal_distribution(img_y_all, y_avg)

                perf = foo_performance(img_y_te, y_avg, thresh)
                
                # CROP
                
                w0, w1, h0, h1 = lst_get[i_fold]()

                y_avg_bin_crop = y_avg_bin[h0:h1, w0:w1]
                clean_crop = img_clean[h0:h1, w0:w1, :]
                
                y_avg_bin_transparent_crop = semi_transparant(clean_crop, y_avg_bin_crop)
                if 0:
                    concurrent([clean_crop, y_avg_bin_crop, y_avg_bin_transparent_crop])
                    
                # Save
                if self.fixed_enc == 0:info_enc = 'Train'
                elif self.fixed_enc == 1:info_enc = 'Fixed'
                elif self.fixed_enc == 2:info_enc = 'FixedTrain'
                
                folder_save = '/scratch/lameeus/data/ghent_altar/output/hierarchy/10_lamb/ifolds'
                filename = f'_enc{info_enc}_ifold{i_fold}_jacc{perf["jaccard"]:.3f}.png'
                # Save y_bin
    
                imsave(os.path.join(folder_save, 'binpred'+filename), y_avg_bin_crop, b_check_duplicate=False)
                
                # Save overlay
                imsave(os.path.join(folder_save, 'overlay'+filename), y_avg_bin_transparent_crop, b_check_duplicate=False)

    def predict_compare_regular(self):
    
        img_y_all = self.k_fold_train_data.get_train_data_all().get_y_train()
        lst_get = [get_borders1, get_borders2, get_borders3, get_borders4, get_borders5, get_borders6]
        img_clean = self.img_x[..., :3]

        for i_fold in range(6):

                y_lst = []
                # Average prediction
                print(f'i_fold = {i_fold}')
                for k in [17, 18, 19]:
                    print(f'k = {k}')
                    for epoch in [36, 37, 38, 39, 40]:
                        print(f'epoch = {epoch}')
                        y = self.predict_regular(i_fold=i_fold, k=k, epoch=epoch)
                        y_lst.append(y)

                y_avg = np.mean(y_lst, axis=0)
                y_avg_bin = self.get_bin(y_avg)

                # Performance
                img_y_te = self.k_fold_train_data.k_split_i(i_fold).get_y_test()

                thresh = optimal_test_thresh_equal_distribution(img_y_all, y_avg)

                perf = foo_performance(img_y_te, y_avg, thresh)

                # CROP

                w0, w1, h0, h1 = lst_get[i_fold]()

                y_avg_bin_crop = y_avg_bin[h0:h1, w0:w1]
                clean_crop = img_clean[h0:h1, w0:w1, :]

                y_avg_bin_transparent_crop = semi_transparant(clean_crop, y_avg_bin_crop)
                if 0:
                    concurrent([clean_crop, y_avg_bin_crop, y_avg_bin_transparent_crop])

                folder_save = '/scratch/lameeus/data/ghent_altar/output/hierarchy/10_lamb/ifolds_regular_tiunet'
                filename = f'_tiunet_ifold{i_fold}_jacc{perf["jaccard"]:.3f}.png'
                # Save y_bin

                imsave(os.path.join(folder_save, 'binpred' + filename), y_avg_bin_crop, b_check_duplicate=False)

                # Save overlay
                imsave(os.path.join(folder_save, 'overlay' + filename), y_avg_bin_transparent_crop,
                       b_check_duplicate=False)

    def get_bin(self, y_pred):
    
        img_y_all = self.k_fold_train_data.get_train_data_all().get_y_train()
    
        thresh = optimal_test_thresh_equal_distribution(img_y_all, y_pred)

        y_pred_bin = np.greater_equal(y_pred[..., 1], thresh)
        return y_pred_bin

    def get_tiunet_preenc(self, k=10, lr=1e-4):
        """

        :param k:
        :param lr:
        :param f_out:
        :return:
        """
    
        model_encoder = self.get_encoder(k)
        n_in = get_mod_n_in(self.mod)
    
        model_tiunet = ti_unet(n_in, filters=k, w=self.w_patch, ext_in=10 // 2, batch_norm=self.batch_norm)
    
        if self.batch_norm:
            o = model_tiunet.get_layer(f'batchnorm_left{self.depth}_0').output
        else:
            o = model_tiunet.get_layer(f'left{self.depth}_0').output
        model_tiunet_encoder = Model(model_tiunet.input, o)
    
        model_tiunet_encoder.set_weights(model_encoder.get_weights())
    
        if self.fixed_enc == 1:
            for layer in model_tiunet_encoder.layers:
                layer.trainable = False
    
        model_tiunet.summary()
    
        compile_segm(model_tiunet, lr=lr)
    
        return model_tiunet
    
    
if __name__ == '__main__':
    Main()
