import os
import numpy as np
import matplotlib.pyplot as plt

from data.datatools import imread, imsave
from data.conversion_tools import annotations2y


def lamb():
    
    folder_in = '/net/gaimfs/research/lameeus/data/Nina/'
    
    im_clean = imread(os.path.join(folder_in, 'clean.png'))
    im_detection = imread(os.path.join(folder_in, 'paintloss_tiunet_enc0.png'))
    
    extra1 = imread('/scratch/lameeus/data/ghent_altar/input/hierarchy/10lamb/mask_pred_comb.png')
    b1 = np.equal(extra1, 255)

    im_update = np.copy(im_clean)

    extra2 =  imread('/home/lameeus/Desktop/extra_annot.png')

    cyan = [0, 255, 255]

    im_update[im_detection.astype(bool), :] = cyan
    
    im_update[b1, :] = cyan

    y_extra2=annotations2y(extra2)
    
    y_extra2_0 = y_extra2[..., 0].astype(bool)
    y_extra2_1 = y_extra2[..., 1].astype(bool)
    im_update[y_extra2_0, :] = im_clean[y_extra2_0, :]
    im_update[y_extra2_1, :] = cyan
    
    plt.imshow(im_update)
    
    from data.conversion_tools import detect_colour
    paintloss_updated = detect_colour(im_update, 'cyan')
    
    imsave('/net/gaimfs/research/lameeus/data/Nina/detection_updated.png', paintloss_updated)
    
    return im_update


if __name__ == '__main__':
    lamb()
