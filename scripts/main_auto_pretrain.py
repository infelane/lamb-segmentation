import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.models import load_model, Model

from data.conversion_tools import img2array, batch2img, img2batch
from data.datatools import pandas_save
from data.preprocessing import rescale0to1
from datasets.default_trainingsets import get_10lamb_old, get_10lamb_6patches, get_13
from main_general import get_training_data
from methods.examples import get_neural_net_ae, neuralNet0
from methods.basic import NeuralNet
from neuralNetwork.optimization import find_learning_rate
from performance.testing import optimal_test_thresh_equal_distribution, test_thresh_incremental
from plotting import concurrent
from preprocessing.image import get_class_weights, get_class_imbalance, get_flow

from neuralNetwork.architectures import ti_unet
from data.modalities import get_mod_n_in
from methods.examples import compile_segm

from scripts.scripts_performance.main_performance import foo_performance


class MainPretrain(object):
    
    fixed_enc = True
    ti = True
    
    """
    You should not need to change the following:
    """
    batch_norm = True
    epochs = 100
    mod = 5
    # Instead of 1e-4 (most likely regular TI-UNet got trained at 1e-3 instead of 1e-4
    # Also used 1e-3 which seems to work nicely. Just gonna do an inbetween for now...
    lr_opt = 3e-4
    
    def init_w(self):
    
        if self.depth == 2:
            self.w_patch = 50
            self.w_ext_in_ae = 28
        elif self.depth == 1:
            self.w_patch = 50
            self.w_ext_in_ae = 12
            self.w_ext_in_ti = 10
            
    def __init__(self,
                 ae_set_nr = None,
                 k=None, depth=None,
                 ti = None,
                 fixed_enc:int=None):
        """
        
        :param k:
        :param depth:
        :param ti:
        :param fixed_enc: 2:
        """

        if ae_set_nr is not None:
            self.ae_set_nr = ae_set_nr
        if k is not None:
            self.k = k
        if depth is not None:
            self.depth = depth
        if ti is not None:
            self.ti = ti
        if fixed_enc is not None:
            assert isinstance(fixed_enc, int)
            self.fixed_enc = fixed_enc
            
        self.init_w()
        
        self.set_img_x()
        
        self.set_flow()
        
        if 1:
            self.train_segm()
            
    def set_img_x(self):
        train_data = get_10lamb_old(self.mod)
        img_x, _, _, _ = get_training_data(train_data)
        # Normalise the input!
        img_x = rescale0to1(img_x)
        self.img_x = img_x
    
    def set_flow(self):
        # w_out should be 2+4*n
        
        w_checker = 512

        if self.ae_set_nr is not None:

            img_x = []
            for set_nr in self.ae_set_nr:

                if set_nr == 10:
                    df = get_10lamb()
                elif set_nr == 13:
                    df = get_13zach()
                elif set_nr == 19:
                    df = get_19hand()

                img_x_i = xy_from_df(df, mod)[0]

                img_x.append()

        else:
            raise NotImplementedError()

            img_x_lst = [self.img_x]

        h_x, w_x = img_x.shape[:2]
        
        x_ae_tr = []
        x_ae_te = []
        for i in range(int(np.ceil(h_x/w_checker))):
            for j in range(int(np.ceil(w_x / w_checker))):
                
                h0 = i*w_checker
                w0 = j*w_checker
                crop_x = img_x[h0:h0+w_checker, w0:w0+w_checker, ...]
                
                if (i + j)%2 == 0:
                    x_ae_tr.append(crop_x)
                else:
                    x_ae_te.append(crop_x)

        self.flow_ae_tr = get_flow(x_ae_tr, x_ae_tr,
                                   w_patch=self.w_patch,
                                   w_ext_in=self.w_ext_in_ae,
                                   )
        self.flow_ae_te = get_flow(x_ae_te, x_ae_te,
                                   w_patch=self.w_patch,
                                   w_ext_in=self.w_ext_in_ae,
                                   )

        """
        Data for Segmentation of 10 lamb
        """
        self.k_fold_train_data = get_10lamb_6patches(5)

        self.flow_segm = get_flow(self.img_x,
                                  self.k_fold_train_data.get_train_data_all().get_y_train(),
                                  w_patch=self.w_patch,
                                  w_ext_in=self.w_ext_in_ti if self.ti else self.w_ext_in_ae,
                                  )
    
    def train_ae(self, epochs=300, verbose=2):
        """
        
        :param epochs: 300 now equals to the epoch the encoder is loaded at for pretraining
        :return:
        """
        from neuralNetwork.optimization import find_learning_rate
        
        k_lst = [self.k]
        
        # Cross entropy
        loss_min = -self.img_x*np.log(np.clip(self.img_x,1e-3,1))-(1-self.img_x)*np.log(np.clip(1-self.img_x,1e-3,1))
        loss_min_mean = np.mean(loss_min)
        loss_min_std = np.std(loss_min)
        print(f'CE loss min = {loss_min_mean} +- {loss_min_std}')
    
        for k in k_lst:
        
            if self.batch_norm:
                info = f'autoencoder_batchnorm/d{self.depth}_k{k}'
            else:
                info = f'autoencoder_v3/d{self.depth}_k{k}'
        
            get_n_ae = lambda lr:get_neural_net_ae(self.mod, k, lr=lr,
                                                   w_in=self.w_patch, depth=self.depth, b_double=False,
                                                   b_split_mod=False, batch_norm = self.batch_norm)
        
            if not self.lr_opt:
                n_ae = get_n_ae(0)
                find_learning_rate(n_ae.get_model(), self.flow_ae_tr, lr1=1e0)
                del n_ae    # Reset to be sure
            n_ae = get_n_ae(self.lr_opt)
            if 0:
                epoch_start = 200
                n_ae.load(os.path.join('/scratch/lameeus/data/ghent_altar/net_weight', info), epoch_start)
            n_ae.train(self.flow_ae_tr, self.flow_ae_te, info=info, epochs=epochs, verbose=verbose)
        
        if verbose == 1:
            self.ae_results(n_ae)
        return n_ae
    
    def train_segm(self):
        folder_save = '/home/lameeus/data/ghent_altar/dataframes'
        
        info_batchnorm = '_batchnorm' if self.batch_norm else ''
        info_fixed = '_encfixed' if self.fixed_enc == 1 else  '_prefixed' if self.fixed_enc == 2 else ''
        info_model = 'tiunet' if self.ti else 'unet'
        
        filename_single = f'pretrained/{info_model}_10lamb_kfold{info_fixed}{info_batchnorm}/d{self.depth}_single'
        path_single = os.path.join(folder_save, filename_single + '.csv')

        get_info = lambda: f'10lamb_kfold_pretrained{info_fixed}{info_batchnorm}/{info_model}_d{self.depth}_k{self.k}_ifold{i_fold}'
        
        img_y_all = self.k_fold_train_data.get_train_data_all().get_y_train()
        
        def get_model():
            if self.ti:
                model = self.get_tiunet_preenc(k=self.k, lr=self.lr_opt)
                
            else:
                model =  self.get_unet_preenc(k=self.k, lr=self.lr_opt)

            if self.fixed_enc == 2:
                n_temp = NeuralNet(model)
    
                folder_weights = '/scratch/lameeus/data/ghent_altar/net_weight'
                folder1 = f'10lamb_kfold_pretrained{"_encfixed"}{info_batchnorm}'
                folder2 = f'{info_model}_d{self.depth}_k{self.k}_ifold{i_fold}'
    
                n_temp.load(os.path.join(folder_weights, folder1, folder2), 100)
    
                del (n_temp)
            
            return model

        w_ext = self.w_ext_in_ti if self.ti else self.w_ext_in_ae
        
        if not self.lr_opt:
            model_segm = get_model()
            find_learning_rate(model_segm, self.flow_segm, lr1=1e0)

        for i_fold in range(6):
            print(f'i_fold = {i_fold}')
            
            model_segm = get_model()
            n_segm = NeuralNet(model_segm, w_ext=w_ext)
            

            train_data_i = self.k_fold_train_data.k_split_i(i_fold)
            img_y_tr = train_data_i.get_y_train()
            img_y_te = train_data_i.get_y_test()
            flow_tr = get_flow(self.img_x, img_y_tr,
                               w_patch=self.w_patch,
                               w_ext_in=w_ext
                               )
            flow_te = get_flow(self.img_x, img_y_te,
                               w_patch=self.w_patch,
                               w_ext_in=w_ext
                               )

            info = get_info()

            for epoch in range(self.epochs):
                n_segm.train(flow_tr, flow_te, epochs=1, verbose=2, info=info)
            
                y_pred = n_segm.predict(self.img_x)
                thresh_single = optimal_test_thresh_equal_distribution(img_y_all, y_pred)
                data_single_i = {'k': self.k,
                                 'i_fold': i_fold,
                                 'epoch': epoch}
                data_single_i.update(foo_performance(img_y_te, y_pred, thresh_single))
                lst_data_single = [data_single_i]
                df_single = pd.DataFrame(lst_data_single)
                pandas_save(path_single, df_single, append=True)
                
        return
    
    def train_y(self):
        
        model_y = self.get_tiunet_y()
        
        return
    
    def ae_results(self, n_ae):
        """
        AE Reconstruction
        """
        
        img_x_ae = n_ae.predict(self.img_x)

        y_denorm = _undo_norm_input(img_x_ae)
        y_clean = y_denorm[..., :3]
        y_rgb = y_denorm[..., 3:6]
        y_ir = y_denorm[..., 6]
        y_irr = y_denorm[..., 7]
        y_xray = y_denorm[..., 8]
        concurrent([y_clean, y_rgb, y_ir, y_irr, y_xray], ['clean', 'rgb', 'ir', 'irr', 'xray'])
        
    def get_encoder(self, k=10, verbose=1, set_info='',
                    epoch_start=300
                    ):
        
        if self.batch_norm:
            info = f'autoencoder_batchnorm{set_info}/d{self.depth}_k{k}'
        else:
            info = f'autoencoder_v3/d{self.depth}_k{k}'
        n_ae = get_neural_net_ae(self.mod, k,
                                 batch_norm=self.batch_norm,
                                 w_in=self.w_patch,
                                 depth=self.depth,
                                 b_double=False,
                                 b_split_mod=False,
                                 verbose=0)
        
        path_weight = os.path.join('/scratch/lameeus/data/ghent_altar/net_weight', info)
        if not os.path.exists(path_weight):
            self.train_ae(verbose=2, epochs=epoch_start)
        
        n_ae.load(path_weight, epoch_start)
        
        # Check if ok
        img_x_ae = n_ae.predict(self.img_x)
        mse = (np.square(img_x_ae - self.img_x)).mean()
        if mse >= .01: print('autoencoder is probably badly trained!')

        model_ae = n_ae.model
        encoder_inputs = model_ae.input
        
        if self.batch_norm:
            encoder_outputs = model_ae.get_layer('batchnorm_enc_output').output
        else:
            encoder_outputs = model_ae.get_layer('encoder_output').output
        
        model_encoder = Model(encoder_inputs, encoder_outputs)
        if verbose: model_encoder.summary()
    
        if self.fixed_enc:
            # make untrainable
            for layer in model_encoder.layers:
                layer.trainable = False
                
        return model_encoder

    def get_unet_preenc(self, k=10, lr=1e-4, f_out=2,
                        ):
        """
        
        :param k:
        :param lr:
        :param f_out:
        :return:
        """
        
        from keras.layers import Conv2D, UpSampling2D, Concatenate, Cropping2D, Conv2DTranspose, BatchNormalization
        from methods.examples import compile_segm
        
        model_encoder = self.get_encoder(k)
        
        b_double = False
        padding = 'valid'

        encoder_outputs = model_encoder.output

        l = encoder_outputs
        
        if self.depth == 2:
            list_w_crop = [12, 4]
        elif self.depth == 1:
            list_w_crop = [4]
        
        for i_d in range(self.depth)[::-1]:
            f = 2 ** i_d * k if b_double else k
            l = Conv2D(f, (3, 3), activation='elu', padding=padding, name=f'dec{i_d+1}')(l)
            
            if self.batch_norm:
                l = BatchNormalization(name=f'batchnorm_dec{i_d+1}')(l)
            
            if 0:
                l = UpSampling2D(2)(l)
            else:
                l = Conv2DTranspose(f, (2, 2), strides=(2, 2))(l)
                if self.batch_norm:
                    l = BatchNormalization(name=f'batchnorm_up{i_d}')(l)
            
            # Combine
            l_left_crop = Cropping2D(list_w_crop[i_d], name=f'crop_enc{i_d}')(
                model_encoder.get_layer(f'enc{i_d}').output)
            l = Concatenate(name=f'conc_dec{i_d}')([l, l_left_crop])

        l = Conv2D(k, (3, 3), activation='elu', padding=padding, name=f'dec{0}')(l)
        if self.batch_norm:
            l = BatchNormalization(name=f'batchnorm_dec{0}')(l)
        decoder_outputs = Conv2D(f_out, (1, 1), activation='softmax', padding=padding)(l)

        model_pretrained_unet = Model(model_encoder.input, decoder_outputs)
        compile_segm(model_pretrained_unet, lr=lr)

        model_pretrained_unet.summary()

        return model_pretrained_unet

    def get_tiunet_preenc(self, k=10, lr=1e-4,
                          set_info='',
                          epoch_start=300
                          ):
        """

        :param k:
        :param lr:
        :param f_out:
        :return:
        """
        
        model_encoder = self.get_encoder(k, set_info=set_info, epoch_start=epoch_start)
        n_in = get_mod_n_in(self.mod)
        
        model_tiunet = ti_unet(n_in, filters=k, w=self.w_patch, ext_in=10//2, batch_norm=self.batch_norm)
        
        if self.batch_norm:
            o = model_tiunet.get_layer(f'batchnorm_left{self.depth}_0').output
        else:
            o = model_tiunet.get_layer(f'left{self.depth}_0').output
        model_tiunet_encoder = Model(model_tiunet.input,o)
        
        model_tiunet_encoder.set_weights(model_encoder.get_weights())
        
        if self.fixed_enc == 1:
            for layer in model_tiunet_encoder.layers:
                layer.trainable = False
                
        model_tiunet.summary()

        compile_segm(model_tiunet, lr=lr)
        
        return model_tiunet

    def get_tiunet_y(self):
        
        from keras.layers import AveragePooling2D
        
        n_in = get_mod_n_in(self.mod)
        model_tiunet = ti_unet(n_in, filters=self.k, w=self.w_patch, ext_in=self.w_ext_in_ti//2, batch_norm=self.batch_norm)
        
        model_ae_example = get_neural_net_ae(self.mod, k=self.k,
                                             w_in=self.w_patch,
                                             b_double=False,
                                             batch_norm=self.batch_norm).get_model()
        
        name_dec_in = 'batchnorm_enc_output' if self.batch_norm else 'encoder_output'
        name_dec_in = f'dec{self.depth}'
        dec_in = model_ae_example.get_layer(name_dec_in).input
        ae_decoder = Model(dec_in, model_ae_example.layers[-1].output)
        
        # TODO use pretrained AE?
        
        name_enc = 'batchnorm_left1_0' if self.batch_norm else 'left1_0'
        layer_enc = model_tiunet.get_layer(name_enc)
        
        # Basically subsample layer (no pooling)
        l = AveragePooling2D((1, 1), (2, 2))(layer_enc.output)
        
        # TODO use crop!
        
        return
        

class MainTransfer(MainPretrain):
    def __init__(self, set_nr, k, i_fold, fixed_enc=0, ti=True, depth=1):

        self.set_nr = set_nr
        self.k = k
        self.i_fold = i_fold
        self.fixed_enc = fixed_enc
        self.ti = ti
        self.depth = depth

        self.init_w()
        self.set_img_x()

        if 0:
            self.train_ae()
        # if 1:
        #     self.pretrain_setencoder()

        if 1:
            self.train_segm()
    
    def set_img_x(self):
        if self.set_nr == 13:
            train_data = get_13(self.mod)
            
            from data.datatools import imread
            from data.conversion_tools import annotations2y
            train_data.y_te = np.copy(train_data.y_tr)
            train_data.y_tr = annotations2y(imread('/home/lameeus/data/ghent_altar/input/hierarchy/13_small/clean_annot_practical.png'),
                                            thresh=.9)
            
            img_x, img_y, _, img_y_te = get_training_data(train_data)
            
        # Normalise the input!
        img_x = rescale0to1(img_x)
        self.img_x = img_x
        self.img_y_tr = img_y
        self.img_y_te = img_y_te

        train_data_10 = get_10lamb_6patches(self.mod).get_train_data_all()
        img_x_10, img_y_10, _, _ = get_training_data(train_data_10)
        # Normalise the input!
        img_x_10 = rescale0to1(img_x_10)

        self.flow_tr_set = get_flow(self.img_x, self.img_y_tr,
                                       w_patch=self.w_patch,
                                       w_ext_in=self.w_ext_in_ti
                                       )
        self.flow_tr_10 = get_flow(img_x_10, img_y_10,
                                       w_patch=self.w_patch,
                                       w_ext_in=self.w_ext_in_ti
                                       )
        n_multiply = 10
        self.flow_tr_set_10 = get_flow([self.img_x]*n_multiply + [img_x_10], [self.img_y_tr]*n_multiply + [img_y_10],
                                       w_patch=self.w_patch,
                                       w_ext_in=self.w_ext_in_ti
                                       )
        
        
        self.flow_ae_tr = get_flow(self.img_x, self.img_x,
                                   w_patch=self.w_patch,
                                   w_ext_in=self.w_ext_in_ae,
                                   )

    def train_ae(self, epochs=100, verbose=2):

        info = f'autoencoder_batchnorm_{self.set_nr}/d{self.depth}_k{self.k}'

        n_ae = get_neural_net_ae(self.mod, self.k, lr=self.lr_opt,
                                                    w_in=self.w_patch, depth=self.depth, b_double=False,
                                                    b_split_mod=False, batch_norm=self.batch_norm)

        n_ae.train(self.flow_ae_tr, info=info, epochs=epochs, verbose=verbose)

        if verbose == 1:
            self.ae_results(n_ae)
        return n_ae
        
    def train_segm(self):
        from figures_paper.overlay import semi_transparant
        
        if self.fixed_enc == -2:
    
            def get_n_model_regular(i_fold=None,
                                    k=None,
                                    epoch=None
                                    ):
        
                n_in = 9
        
                model_tiunet = ti_unet(n_in, filters=k,
                                       w=self.w_patch,
                                       ext_in=10 // 2,
                                       batch_norm=True,
                                       wrong_batch_norm=True)
                compile_segm(model_tiunet, 1e-4)
        
                """ TODO wrong tiunet """
        
                n = NeuralNet(model_tiunet, w_ext=10)
        
                info = f'10lamb_kfold/ti_unet_k{k}_kfold{i_fold}'
                folder = os.path.join('/scratch/lameeus/data/ghent_altar/net_weight', info)
                n.load(folder=folder, epoch=epoch)
        
                return n
            
            n_segm = get_n_model_regular(i_fold=self.i_fold, k=self.k, epoch=40)
        
        elif self.fixed_enc == -1:
            # No init
            model_segm = self.get_tiunet_preenc(k=self.k, lr=self.lr_opt)
            n_segm = NeuralNet(model_segm, w_ext=self.w_ext_in_ti)
        
        elif self.fixed_enc in [0, 1, 2]:
            
            # Load model
            model_segm = self.get_tiunet_preenc(k=self.k, lr=self.lr_opt)
            n_segm = NeuralNet(model_segm, w_ext=self.w_ext_in_ti)
        
            # Train on set
            folder_weights = '/scratch/lameeus/data/ghent_altar/net_weight'
            if self.fixed_enc == 0:
                folder1 = '10lamb_kfold_pretrained_batchnorm'
            elif self.fixed_enc == 1:
                folder1 = '10lamb_kfold_pretrained_encfixed_batchnorm'
            elif self.fixed_enc == 2:
                folder1 = '10lamb_kfold_pretrained_prefixed_batchnorm'
                
            folder2 = f'{"tiunet"}_d{self.depth}_k{self.k}_ifold{self.i_fold}'
            
            n_segm.load(os.path.join(folder_weights, folder1, folder2), 100)
        
        elif self.fixed_enc == 3:
    
            model_segm = self.get_tiunet_preenc(k=self.k, lr=self.lr_opt, set_info=f'_{self.set_nr}', epoch_start=100)
            n_segm = NeuralNet(model_segm, w_ext=self.w_ext_in_ti)
            
        else:
            NotImplementedError()
        
        def foo(n_segm, b=0):
            y_pred = n_segm.predict(self.img_x)
    
            thresh_single = optimal_test_thresh_equal_distribution(self.img_y_te, y_pred)
            # data_single_i = {'k': self.k,
            #                  'i_fold': i_fold,
            #                  'epoch': epoch}
            print(foo_performance(self.img_y_te, y_pred, thresh_single))

            img_clean = self.img_x[..., :3]
            concurrent([img_clean,
                        y_pred[..., 1],
                        y_pred[..., 1] >= thresh_single,
                        semi_transparant(img_clean, y_pred[..., 1] >= thresh_single)
                        ])
            
            if b:
                from data.datatools import imsave
                
                folder = '/home/lameeus/data/ghent_altar/output/hierarchy/'
                info_epoch = f'_epoch{n_segm.epoch}' if n_segm.epoch > 0 else ''
                filename = folder + f'13_small/pred_transfer_kfoldenc{self.fixed_enc}_ifold{self.i_fold}_avg{info_epoch}.png'
                imsave(filename,
                       y_pred[..., 1])

        def set_encoder_state(model, trainable=False):
    
            assert len(model.layers) == 14
            for layer in model.layers[:7]:
                layer.trainable = trainable
            compile_segm(model)
       
        n_segm.epoch = 0
        
        # Without pretraining
        foo(n_segm)
        
        # epochs
        
        set_encoder_state(n_segm.model, trainable=False)
        
        for _ in range(10):
            n_segm.train(self.flow_tr_10, epochs=1, verbose=2)

        foo(n_segm, 0)
        
        if 0:
            set_encoder_state(n_segm.model, trainable=True)
            
            for _ in range(1):
                n_segm.train(self.flow_tr_set_10, epochs=10, verbose=2)
                foo(n_segm, 0)
    
        set_encoder_state(n_segm.model, trainable=True)

        for _ in range(10):
            n_segm.train(self.flow_tr_set, epochs=1, verbose=2)
            
        foo(n_segm, 0)
                
    
def main():
    """

    :return:
    """
    
    ### Settings
    mod = 5

    w_patch = 16*2

    """
    Data (all important modalities)
    """

    # folder_windows = r'C:\Users\Laurens_laptop_w\OneDrive - UGent\data\10lamb'
    train_data = get_10lamb_old(mod)
    img_x, img_y_tr, _, _ = get_training_data(train_data)
    # Normalise the input!
    img_x = rescale0to1(img_x)
    
    """
    Train segmentation
        1) reuse everything
        2) fix encoder
    """
    
    if 1:
        
        if 1:
            b_encoder_fixed = False
    
            info_enc_fixed = '_enc_fixed' if b_encoder_fixed else ''
            get_info = lambda: f'10lamb_kfold_pretrained{info_enc_fixed}/unet_enc_k{k}_ifold{i_fold}'
        
            n_epochs = 40
            
            k = 10
            
            if k == 10:
                epoch_w = 100
            else:
                raise NotImplementedError()

            ### Settings you don't have to change:

            w_patch = 50
            w_ext_in = 28
            b_double = False
            padding = 'valid'
            
            # TODO flag for converting encoder to dilated conv
            
            def get_unet_pretrained_encoder():
    
                model_encoder = get_model_encoder()

                encoder_inputs = model_encoder.input
                
                decoder_outputs = decoder(model_encoder, f_out=2)
    
                model_pretrained_unet = Model(encoder_inputs, decoder_outputs)
                from methods.examples import compile_segm
                compile_segm(model_pretrained_unet, lr=1e-4)

                model_pretrained_unet.summary()
                
                return model_pretrained_unet
                
            """
            Train
            """
        
            k_fold_train_data = get_10lamb_6patches(5)
            for i_fold in range(6):
                
                """
                Get a new network (not trained yet for segmentation)
                """
                
                model_pretrained_unet = get_unet_pretrained_encoder()
                n_pretrained_unet = NeuralNet(model_pretrained_unet)
    
                """
                The data
                """
    
                train_data_i = k_fold_train_data.k_split_i(i_fold)

                info = get_info()

                img_y_tr = train_data_i.get_y_train()
                img_y_te = train_data_i.get_y_test()
    
                flow_tr = get_flow(img_x, img_y_tr,
                                   w_patch=w_patch,  # Comes from 10
                                   w_ext_in=w_ext_in
                                   )
    
                flow_te = get_flow(img_x, img_y_te,
                                   w_patch=w_patch,  # Comes from 10
                                   w_ext_in=w_ext_in
                                   )

                n_pretrained_unet.train(flow_tr, flow_te, epochs=n_epochs, verbose=1, info=info)
            
                """
                Prediction
                """
    
                n_pretrained_unet.w_ext = w_ext_in
                y_pred = n_pretrained_unet.predict(img_x)
                
                concurrent([y_pred[..., 1]])
            
    """
    Classification
    """
    
    if 1:
        im_clean = img_x[..., :3]
        
        k = 8
        i_fold = 3
        epoch_last = 40
    
        from methods.examples import kappa_loss, weighted_categorical_crossentropy
        from performance.metrics import accuracy_with0, jaccard_with0
        loss = weighted_categorical_crossentropy((1, 1))
        
        list_y_pred = []
        
        ### K fold validation
        k_fold_train_data = get_10lamb_6patches(5)
        train_data_i = k_fold_train_data.k_split_i(i_fold)
        img_y_tr = train_data_i.get_y_train()
        img_y_te = train_data_i.get_y_test()
        
        for epoch in np.arange(31, epoch_last+1):
            filepath_model = f'/scratch/lameeus/data/ghent_altar/net_weight/10lamb_kfold/ti_unet_k{k}_kfold{i_fold}/w_{epoch}.h5'
    
            model = load_model(filepath_model, custom_objects={
                                                               'loss': loss,
                                                               'accuracy_with0': accuracy_with0,
                                                               'jaccard_with0': jaccard_with0,
                                                               'kappa_loss': kappa_loss
                                                               })
            
            n = NeuralNet(model, w_ext=10)
            y_pred = n.predict(img_x)
            
            list_y_pred.append(y_pred)
        
        y_pred_mean = np.mean(list_y_pred, axis=0)
        q1 = y_pred_mean[..., 1]
        concurrent([q1, q1.round(), im_clean])
        
        """
        Optimal threshold (making conf matrix symmetric, not based on maximising kappa)
        """
        y_gt = np.any([img_y_tr, img_y_te], axis=0)

        from performance.testing import _get_scores, filter_non_zero
        def foo_performance(y_true, y_pred, thresh):
            # is basically argmax
            y_pred_thresh_arg = np.greater_equal(y_pred[..., 1], thresh)
    
            y_true_flat, y_pred_thresh_arg_flat = filter_non_zero(y_true, y_pred_thresh_arg)
            y_te_argmax = np.argmax(y_true_flat, axis=-1)
    
            # Kappa
            return _get_scores(y_te_argmax, y_pred_thresh_arg_flat)[-1]
        
        """
        1. BEST? PERFORMANCE based on test set
        """
        
        print('1. Test distribution optimization')
        
        thresh = optimal_test_thresh_equal_distribution(img_y_te, y_pred_mean)
        q1_thresh = np.greater_equal(q1, thresh)
        concurrent([q1, q1_thresh, im_clean])
        
        print(f'thresh: {thresh}')
        
        # Test, train, both
        print('Kappa performance:')
        print('\ttrain:', foo_performance(img_y_tr, y_pred_mean, thresh))
        print('\ttestset:', foo_performance(img_y_te, y_pred_mean, thresh))
        print('\tboth:', foo_performance(y_gt, y_pred_mean, thresh))

        print('\nIncremental optimization on test set')

        test_thresh2 = test_thresh_incremental(y_pred_mean, img_y_tr, img_y_te, n=5,
                                               verbose=0)

        print('Kappa performance:')
        print('\ttrain:', foo_performance(img_y_tr, y_pred_mean, test_thresh2))
        print('\ttestset:', foo_performance(img_y_te, y_pred_mean, test_thresh2))
        print('\tboth:', foo_performance(y_gt, y_pred_mean, test_thresh2))
        
        """
        2. based on train
        """
        
        print('\n2. Training distribution optimization')
        
        thresh = optimal_test_thresh_equal_distribution(img_y_tr, y_pred_mean)
        q1_thresh = np.greater_equal(q1, thresh)
        concurrent([q1, q1_thresh, im_clean])
        
        print(f'thresh: {thresh}')
        
        # Test, train, both
        print('Kappa performance:')
        print('\ttrain:', foo_performance(img_y_tr, y_pred_mean, thresh))
        print('\ttestset:', foo_performance(img_y_te, y_pred_mean, thresh))
        print('\tboth:', foo_performance(y_gt, y_pred_mean, thresh))
        
        """
        3. CONSISTENT: based on train+set
        """

        print('\n3. all GT distribution optimization')

        thresh = optimal_test_thresh_equal_distribution(y_gt, y_pred_mean)
        q1_thresh = np.greater_equal(q1, thresh)
        concurrent([q1, q1_thresh, im_clean])
        
        print(f'thresh: {thresh}')
        
        # Test, train, both
        print('Kappa performance:')
        print('\ttrain:', foo_performance(img_y_tr, y_pred_mean, thresh))
        print('\ttestset:', foo_performance(img_y_te, y_pred_mean, thresh))
        print('\tboth:', foo_performance(y_gt, y_pred_mean, thresh))
        
        if 0:
            """
            4. DUMB/Not needed: Based on prediction of whole panel
            """
            
            thresh = optimal_test_thresh_equal_distribution(y_gt, y_pred_mean, mask_true=False)
            q1_thresh = np.greater_equal(q1, thresh)
            concurrent([q1, q1_thresh, im_clean])
    
    print('Done')


def _undo_norm_input(x_norm):
    # undo the normalisation of the input
    return np.clip((x_norm)*255., 0, 255).astype(np.uint8)


if __name__ == '__main__':

    if 0:
        for depth in [1]:
            for k in range(2, 20):

                print(f'k = {k}')
                try:
                    Main(k, depth, ti=True, fixed_enc=2)
                except Exception:
                    print('fail1')

    """
    fixed_enc = -1: No initialisation
    """
    if 0:
        MainTransfer(set_nr=13,
                     k=12,
                     i_fold=0,
                     fixed_enc=2,
                     ti=True,
                     depth=1
                     )

    if 1:
        # TODO Pretrain 2 pooling layers, with split
        # AE trained on 10, 13 and 19.
        MainPretrain(ae_set_nr=[10, 13, 19],
                     k=12,
                     ti=True,
                     fixed_enc=0,
                     depth=2
                     )

    # for depth in [1]:
    #     for k in range(15, 31):
    #         # Ignore exceptions for now
    #
    #         print(f'k = {k}')
    #         try:
    #             MainPretrain(k, depth, ti = True, fixed_enc=True)
    #         except Exception: print('fail1')
    #
    #         try:
    #             MainPretrain(k, depth, ti=True, fixed_enc = False)
    #         except Exception: print('fail3')
            
            # try:
            #     MainPretrain(k, depth, ti = False, fixed_enc=True)
            # except Exception:print('fail2')

            # try:
            #     MainPretrain(k, depth, ti=False, fixed_enc = False)
            # except Exception: print('fail4')
    
    # MainPretrain(16, depth=1)
    # MainPretrain(10, depth=2)
    # main()
