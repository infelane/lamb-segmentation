import os

import matplotlib.pyplot as plt
import numpy as np

# from keras.models import load_model, Model
#
# # from data.conversion_tools import img2array, batch2img, img2batch
# from data.preprocessing import rescale0to1
from datasets.default_trainingsets import get_10lamb_old, get_10lamb_6patches
# from datasets.script_combine_10lamb_6annot import get_borders1, get_borders2, get_borders3, get_borders4, get_borders5, get_borders6
from main_general import get_training_data
# # from methods.examples import get_neural_net_ae, neuralNet0
# from methods.basic import NeuralNet
# # from neuralNetwork.optimization import find_learning_rate
# from performance.testing import optimal_test_thresh_equal_distribution, test_thresh_incremental
from plotting import concurrent
# # from preprocessing.image import get_class_weights, get_class_imbalance, get_flow
from figures_paper.overlay import semi_transparant
# from scripts.scripts_performance.main_performance import foo_performance
#
# from neuralNetwork.architectures import ti_unet
# from data.modalities import get_mod_n_in
# from methods.examples import compile_segm

from data.conversion_tools import annotations2y
from data.datatools import imsave, imread
from datasets.script_combine_10lamb_6annot import get_borders1, get_borders2, get_borders3, get_borders4, get_borders5, get_borders6


def folds_annot():
    train_data = get_10lamb_old(5)
    img_x, _, _, _ = get_training_data(train_data)
    
    img_clean = img_x[..., :3]

    lst_get = [get_borders1, get_borders2, get_borders3, get_borders4, get_borders5, get_borders6]

    for i_fold in range(6):
        
        img_annot = imread(f'/home/lameeus/data/ghent_altar/input/hierachy/10_lamb/annotations/kfold/annot_{i_fold+1}.png')

        y1 = annotations2y(img_annot, thresh=.8)[..., 1]

        a = semi_transparant(img_clean, y1.astype(bool))
    
        w0, w1, h0, h1 = lst_get[i_fold]()
        clean_annot_crop = a[h0:h1, w0:w1, :]

        img_clean_crop = img_clean[h0:h1, w0:w1, :]
        
        if 0: concurrent([img_clean_crop, clean_annot_crop])
        
        folder_save = '/scratch/lameeus/data/ghent_altar/input/hierarchy/10lamb/ifolds'

        imsave(os.path.join(folder_save, f'clean_crop_ifold{i_fold}.png'), img_clean_crop)
        imsave(os.path.join(folder_save, f'clean_annot_crop_ifold{i_fold}.png'), clean_annot_crop)
        
        pass
        
    
def combine_stitches():
    
    folder = '/home/lameeus/data/ghent_altar/output/hierarchy/10_lamb/inpainting'
    im_base = imread(os.path.join(folder, 'inpainting_stitch_f2_g3_fixed_v0_linear_c256.png'))
    im_leftbot = imread(os.path.join(folder, 'inpainting_stitch_f2_g3_fixed_v1_linear_c256_manual.png'))
    
    im_base[3*256:, :(3)*256, :] = im_leftbot[3*256:, :(3)*256, :]
    
    if 0:
        plt.imshow(im_base)
        imsave(os.path.join(folder, 'inpainting_comb.png'), im_base)

    from figures_paper.overlay import semi_transparant
    im_clean = imread('/home/lameeus/data/ghent_altar/input/hierarchy/10_lamb/clean.png')
    im_paintloss = imread('/home/lameeus/data/ghent_altar/output/hierarchy/10_lamb/detection_updated.png')
    im_overlay = semi_transparant(im_clean, im_paintloss)

    if 1:
        plt.imshow(im_overlay)
        folder = '/home/lameeus/data/ghent_altar/output/hierarchy/10_lamb/fancy'
        imsave(os.path.join(folder, 'det_overlay.png'), im_overlay)


def continues_learning():
    folder = '/home/lameeus/data/ghent_altar/input/hierarchy/13_small'
    im_clean = imread(os.path.join(folder, 'clean.png'))[..., :3]
    im_annot0 = imread(os.path.join(folder, 'annot.tif'))
    im_annot1 = imread(os.path.join(folder, 'clean_annot_practical.png'))

    y_true = annotations2y(im_annot0)
    y_true_extra = annotations2y(im_annot1, thresh=.9)
    
    folder = '/home/lameeus/data/ghent_altar/output/hierarchy/13_small/practical_annotations'
    y_pred0 = imread(os.path.join(folder, 'pred_transfer_kfoldenc2_ifold0_avg.png'))
    
    folder = '/home/lameeus/data/ghent_altar/output/hierarchy/13_small'
    y_pred1 = imread(os.path.join(folder, 'pred_transfer_kfoldenc2_ifold0_avg_epoch50_J0427.png'))
    
    from performance.testing import optimal_test_thresh_equal_distribution
    from scripts.scripts_performance.main_performance import foo_performance
    from figures_paper.overlay import semi_transparant
    def get_bin(y_pred):

        assert len(y_pred.shape) == 2
        y_pred01 = np.stack([1-y_pred, y_pred], axis=-1)
        thresh = optimal_test_thresh_equal_distribution(y_true, y_pred01)

        print(foo_performance(y_true, y_pred01, thresh))

        y_pred_bin = y_pred >= thresh
        
        return y_pred_bin

    y_pred0_bin = get_bin(y_pred0)
    y_pred1_bin = get_bin(y_pred1)

    y_pred0_bin_fancy = semi_transparant(im_clean, y_pred0_bin)
    y_pred1_bin_fancy = semi_transparant(im_clean, y_pred1_bin)
    
    concurrent([im_clean, y_true[..., 0], y_true_extra[..., 0], y_pred0, y_pred1, y_pred0_bin, y_pred1_bin,
                y_pred0_bin_fancy,
                y_pred1_bin_fancy])
    
    folder_save = '/home/lameeus/data/ghent_altar/output/hierarchy/13_small/fancy'
    imsave(os.path.join(folder_save, 'overalytrain10.png'), y_pred0_bin_fancy)
    imsave(os.path.join(folder_save, 'overalytrain10train13.png'), y_pred1_bin_fancy)
    
    return
    

def class_imbalance():
    
    y = annotations2y(imread('/home/lameeus/data/ghent_altar/input/hierarchy/10_lamb/annotations/kfold/annot_comb.png'))

    n = np.sum(y, axis=(0, 1))
    print(n)
    print('f:', n/np.sum(n))

    return


if __name__ == '__main__':
    # folds_annot()
    # combine_stitches()
    # continues_learning()
    class_imbalance()
    