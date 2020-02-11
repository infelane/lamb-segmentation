import numpy as np
import matplotlib.pyplot as plt

from data.example_splits import panel19withoutRightBot
from data.conversion_tools import annotations2y, y2bool_annot
from data.modalities import get_mod_set
from data.preprocessing import img2array, batch2img
from datasets.examples import get_19hand
from methods.basic import Threshholding, local_thresholding
from plotting import concurrent
from methods.examples import neuralNet0
from neuralNetwork.optimization import find_learning_rate
from performance.testing import test, filter_non_zero, get_y_pred_thresh
from preprocessing.image import get_class_weights, get_class_imbalance


def get_training_data(train_data):
    
    x = train_data.get_x_train()
    y_tr = train_data.get_y_train()
    
    x_te = train_data.get_x_test()
    y_te = train_data.get_y_test()
    
    return x, y_tr, x_te, y_te


if __name__ == '__main__':
    ### Settings
    
    mod = 5    # 'clean'   #'all'; 5 (everything except UVF)
    b_imbalance = True
    
    verbose = 0

    epochs = 40 # was 10
    b_opt_lr = False # TODO watch out for flag setting
    
    ## amount of filters
    k_lst = [1, 2, 4, 8, 16, 32]
    # k_lst = [1, 2, 4, 8, 16, 32, 64, 128]
    # k_lst = [16, 32, 64]    # 3x3 conv
    # k_lst = np.arange(16, 52, 2)
    k_lst = np.arange(1, 21, 1)
    
    # k_lst = [5] # Test new class imbalance
    
    ### Data
    if 0:
        a = get_19hand()
        b = False
        if b:
            a.plot()

        ### Training/Validation data
        img_y = a.get('annot')
        y = annotations2y(img_y)
        y_annot = y2bool_annot(y)

        b = False
        if b:
            y_annot_tr, y_annot_te = panel19withoutRightBot(y_annot)
        
            concurrent([a.get('clean'), y_annot, y_annot_tr, y_annot_te],
                       ['clean', 'annotation', 'train annot', 'test annot'])

    from datasets.training_examples import get_train19_topleft, get_13botleftshuang

    if 0:
        train_data = get_train19_topleft(mod=mod)
    else:
        train_data = get_13botleftshuang(mod=mod)

    # TODO normalise inputs This seems to be super important...
    # train_data.x = (1/255. * train_data.x).astype(np.float16)
    # train_data.x = (255. * train_data.x).astype(np.float16)

    x, y_tr, x_te, y_te = get_training_data(train_data)

    from preprocessing.image import get_flow

    # To get w_ext
    w_ext = neuralNet0(mod=mod, k=1, verbose=1).w_ext

    flow_tr = get_flow(x[0], y_tr[0],
                       w_patch=10,  # Comes from 10
                       w_ext_in=w_ext
                       )
    
    flow_te = get_flow(x_te[0], y_te[0],
                       w_patch=10,  # Comes from 10
                       w_ext_in=w_ext
                       )

    b = 1

    class_weight = (1, 1)
    if b:
        # Balance the data
        class_weight_tr = get_class_weights(flow_tr)
        class_weight = tuple(c_i * c_j  for c_i, c_j  in zip(class_weight, class_weight_tr))
    
    if b_imbalance:
        # Introduce class imbalance to let the network train there is class imbalance.
        
        class_imbalance_te = get_class_imbalance(flow_te)
        
        b_geometric_mean = False
        if b_geometric_mean:
            # Act as if class imbalance (n1/n0) is only (n1/n0)**.5
            # or (f1'/f0') is only (f1/f0)**.5
        
            def f_i_geometric_mean(f_i):
                return 1/((1/f_i - 1)**.5 + 1)
        
            geometric_class_imbalance = tuple(map(f_i_geometric_mean, class_imbalance_te))
            class_weight_geometric = tuple(2. * f_i for f_i in geometric_class_imbalance)
        
        else:
            """
            Introduce class imbalance through the weights
            """
            class_weight_geometric = tuple(2. * f_i for f_i in class_imbalance_te)
    
        class_weight = tuple(c_i * c_j for c_i, c_j in zip(class_weight, class_weight_geometric))
    
    print(f'final class_weight: {class_weight}')
    
    for k in k_lst:
        print(f'\n\tk = {k}')
    
        n = neuralNet0(mod=mod, k=k, verbose=verbose, class_weights=class_weight)
        
        if b_opt_lr:
            ### Finding optimal lr

            lr_opt = find_learning_rate(n.get_model(), flow_tr, class_weight, verbose=verbose)
        else:
    
            lr_opt = 1e-0   # Fully connected
            lr_opt = 1e-1   # CNN batchnorm
            lr_opt = 1e0    # ti unet
            lr_opt = 1e-3   # ti unet + NADAM
            lr_opt = 5e-3   # ti unet + NADAM (class imbalanced!)

        print(f'Optimal expected learning rate: {lr_opt}')
        
        n = neuralNet0(mod=mod, lr=lr_opt, k=k, verbose=verbose, class_weights=class_weight)
        info = f'ti_unet_k{k}'
        if b_imbalance:
            info += '_imbalanced'
        n.train(flow_tr, flow_te, epochs=epochs, verbose=verbose, info=info)
        
        b = False
        if b:
            # Model
            t = Threshholding()
            
            t.method = local_thresholding
            
            o = t.predict(a.get('clean'))
        else:
            x_img = img2array(x)
        
            y_pred = n.predict(x_img)
            o = y_pred[..., 1]
        
        b = False
        if b:
            # plotting results
            concurrent([a.get('clean'), o], ['clean', 'prediction'])
        
        ### Evaluation

        if 1:
            test_thresh = test(y_pred, y_tr, y_te, verbose=verbose, d_thresh=.01 )
        else:
            test_thresh = .96
            
        o2 = np.greater_equal(o, test_thresh)

        if 0:
            concurrent([a.get('clean'), o, o2], ['clean', 'prediction', f'thresh {test_thresh}'], verbose=verbose)
    
        ### test class imbalance

        y_pred_img, y_te_img = map(batch2img, (y_pred, y_te))
        y_te_filter, y_pred_filter = filter_non_zero(y_te_img, y_pred_img)
        
        def class_distribution(y):
            
            assert len(y.shape) == 2, y.shape
    
            n_01 = np.sum(y, axis=0)
            f_01 = n_01 / sum(n_01)
            
            return f_01

        f_01_te = class_distribution(y_te_filter)
        f_01_pred_50 = class_distribution(get_y_pred_thresh(y_pred_filter, thresh=.5))
        f_01_pred_85 = class_distribution(get_y_pred_thresh(y_pred_filter, thresh=.85))
       
        d_thresh = 0.01
        thresh_lst = np.arange(d_thresh, 1, d_thresh)
        f_01_lst = [class_distribution(get_y_pred_thresh(y_pred_filter, thresh=thresh_i)) for thresh_i in thresh_lst]
        
        f_1_lst = list(zip(*f_01_lst))[1]
        
        if 0:
            plt.figure()
            plt.plot(thresh_lst, f_1_lst)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title('Predicted class distribution versus prediction threshold')
        
    plt.show()
    
    print('Done')
