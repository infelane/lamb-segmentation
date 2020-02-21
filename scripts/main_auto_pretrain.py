import os

import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model, Model

from data.conversion_tools import img2array, batch2img, img2batch
from data.preprocessing import rescale0to1
from datasets.default_trainingsets import get_10lamb_all, get_10lamb_6patches
from main_general import get_training_data
from methods.examples import get_neural_net_ae, neuralNet0
from methods.basic import NeuralNet
from neuralNetwork.optimization import find_learning_rate
from performance.testing import optimal_test_thresh_equal_distribution, test_thresh_incremental
from plotting import concurrent
from preprocessing.image import get_class_weights, get_class_imbalance, get_flow


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
    train_data = get_10lamb_all(mod)
    img_x, img_y_tr, _, _ = get_training_data(train_data)
    # Normalise the input!
    img_x = rescale0to1(img_x)
    
    """
    Train autoencoder
    2) combine modalities
    1) split modalities
    """
    
    if 0:
        # Settings:
        k_lst = np.arange(1, 50)
        epochs = 100

        # w_out should be 2+4*n
        w_patch = 50
        w_ext_in = 28
        
        if 1:
            # Split image in half, or like in 4 squares and 2/4 is train, 2/4 test.
            
            h, w = img_x.shape[:2]
            x11 = img_x[:h//2, :w//2, :]
            x12 = img_x[:h//2, w//2:, :]
            x21 = img_x[h//2:, :w//2, :]
            x22 = img_x[h//2:, w//2:, :]
        
            flow_ae_tr = get_flow([x11, x22], [x11, x22],
                                  w_patch=w_patch,
                                  w_ext_in=w_ext_in,
                                  )
            flow_ae_te = get_flow([x12, x21], [x12, x21],
                                  w_patch=w_patch,
                                  w_ext_in=w_ext_in,
                                  )
        
        for k in k_lst:
            for b_double in [True, False]:
                for b_split_mod in [True, False]:
                    info = f'autoencoder_v2/k{k}'
                    if b_double: info+= '_doubling'
                    if b_split_mod: info += '_modsplit'
                    
                    n = get_neural_net_ae(mod, k, w_in=w_patch, b_double=b_double, b_split_mod=b_split_mod)
     
                    n.train(flow_ae_tr, flow_ae_te, info=info, epochs=epochs)

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

            
            def get_model_encoder():
                """
                
                :return:
                """

                """
                Take the auto-encoder
                """

                n = get_neural_net_ae(mod, k,
                                      w_in=w_patch,
                                      b_double=False, b_split_mod=False
                                      )
                model_ae = n.model

                sub_folder = f'k{k}'  # No doubling, no modsplit

                path_weights = os.path.join('/scratch/lameeus/data/ghent_altar/net_weight/autoencoder_v2', sub_folder,
                                            f'w_{epoch_w}.h5')
                model_ae.load_weights(path_weights)

                # Check if ok
                img_x_ae = n.predict(img_x)
                mse = (np.square(img_x_ae - img_x)).mean()
                if mse >= .1: print('autoencoder is probably shitty trained!')
    
                """
                Get the encoder
                """
                
                encoder_inputs = model_ae.input
                encoder_outputs = model_ae.get_layer('encoder_output').output
    
                model_encoder = Model(encoder_inputs, encoder_outputs)
                if 0: model_encoder.summary()
    
                if b_encoder_fixed:
                    # make untrainable
                    for layer in model_encoder.layers:
                        layer.trainable = False
            
                return model_encoder
             
            """
            Construct decoder
            """

            from keras.layers import Conv2D, UpSampling2D, Concatenate, Cropping2D
            def decoder(model_encoder, f_out):
                
                encoder_outputs = model_encoder.output
        
                f = 2 ** 1 * k if b_double else k
                l = Conv2D(f, (3, 3), activation='elu', padding=padding)(encoder_outputs)
    
                l = UpSampling2D(2)(l)
                # Combine
                l_left_crop = Cropping2D(4)(model_encoder._layers_by_depth[2][0].output)
                l = Concatenate()([l, l_left_crop])
    
                f = 2 ** 0 * k if b_double else k
                l = Conv2D(f, (3, 3), activation='elu', padding=padding)(l)
    
                l = UpSampling2D(2)(l)
                
                l_left_crop = Cropping2D(12)(model_encoder._layers_by_depth[4][0].output)
                l = Concatenate()([l, l_left_crop])
    
                outputs = Conv2D(f_out, (3, 3), activation='sigmoid', padding=padding)(l)
    
                return outputs
            
            # TODO flag for converting encoder to dilated conv
            
            def get_unet_pretrained_encoder():
    
                model_encoder = get_model_encoder()

                encoder_inputs = model_encoder.input
                
                decoder_outputs = decoder(model_encoder, f_out=2)
    
                model_pretrained_unet = Model(encoder_inputs, decoder_outputs)
                from methods.examples import compile0
                compile0(model_pretrained_unet, lr=1e-3)

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
            
        if 0:
        
            epoch_w = 10
            
            n = get_neural_net_ae(mod, k)
            model_ae = n.model
            path_weights = f'/scratch/lameeus/data/ghent_altar/net_weight/autoencoder/k{k}/w_{epoch_w}.h5'
            model_ae.load_weights(path_weights)
            
            if 0:
                # Test prediction
                n0 = NeuralNet(model_ae)
                y0 = n0.predict(img_x, w=w_patch)
                concurrent([_undo_norm_input(y0)[..., :3]], ['clean'])
    
            encoder_inputs = model_ae.input
            encoder_outputs = model_ae.get_layer('encoder_output').output
            
            model_encoder = Model(encoder_inputs, encoder_outputs)
            model_encoder.summary()
            
            # make untrainable
            for layer in model_encoder.layers:
                layer.trainable = False
    
            model_encoder.summary()
    
            from keras.layers import Conv2D, UpSampling2D
            
            l = Conv2D(9*2, (3, 3), activation='elu', padding='same')(encoder_outputs)
            l = UpSampling2D(2)(l)
            l = Conv2D(9*1, (3, 3), activation='elu', padding='same')(l)
            l = UpSampling2D(2)(l)
            segm_output = Conv2D(2, (3, 3), activation='softmax', padding='same')(l)
            
            model_segm = Model(encoder_inputs, segm_output)
            model_segm.summary()
            
            from methods.examples import compile0
            compile0(model_segm, lr=1e-3)
    
            n_segm = NeuralNet(model_segm)
    
            flow_segm = get_flow(img_x, img_y_tr,
                               w_patch=w_patch,
                               w_ext_in=0
                               )
    
            n_segm.train(flow_segm, info='segm_aepretrain', epochs=10)
    
            # TODO ae_flow_test (just split image in half, or like in 4 squares and 2/4 is train, 2/4 test.
    
            y_segm = n_segm.predict(img_x, w=w_patch*10)
        
            concurrent([_undo_norm_input(img_x[..., :3]), y_segm[..., 1], img_y_tr[..., 0], img_y_tr[..., 1]])
            
    """
        3) without pretraining
    """
    if 0:
        ### Settings
        
        k_min, k_max = 9, 30
        epochs = 40
        arch = 'ti_unet'
        
        ###
        
        k_lst = np.arange(k_min, k_max+1)
        
        k_fold_train_data = get_10lamb_6patches(5)

        w_ext_in = neuralNet0(mod=mod, k=1, verbose=1).w_ext
        
        for k in k_lst:

            for i_fold in range(6):
                train_data_i = k_fold_train_data.k_split_i(i_fold)
                
                img_y_tr = train_data_i.get_y_train()
                img_y_te = train_data_i.get_y_test()
    
                flow_tr = get_flow(img_x, img_y_tr,
                                   w_patch=10,  # Comes from 10
                                   w_ext_in=w_ext_in
                                   )
    
                flow_te = get_flow(img_x, img_y_te,
                                   w_patch=10,  # Comes from 10
                                   w_ext_in=w_ext_in
                                   )
    
                n = neuralNet0(mod=mod, k=k, verbose=1)
                if 0:
                    lr_opt = find_learning_rate(n.get_model(), flow_tr, verbose=1)
                else:
                    lr_opt = 1e-3
                
                n = neuralNet0(mod=mod, lr=lr_opt, k=k, verbose=1)
    
                info = f'10lamb_kfold/{arch}_k{k}_kfold{i_fold}'
                
                n.train(flow_tr, flow_te, epochs=epochs, verbose=1, info=info)
            
    """
    Prediction
    """
    
    def foo1(n_w=1):

        y = n.predict(img_x, w=w_patch*n_w)
    
        y_denorm = _undo_norm_input(y)
        y_clean = y_denorm[..., :3]
        y_rgb = y_denorm[..., 3:6]
        y_ir = y_denorm[..., 6]
        y_irr = y_denorm[..., 7]
        y_xray = y_denorm[..., 8]
        concurrent([y_clean, y_rgb, y_ir, y_irr, y_xray], ['clean', 'rgb', 'ir', 'irr', 'xray'])
    
    if 0:
        foo1(1)
        
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
    main()
