import os

import matplotlib.pyplot as plt
import numpy as np

from data.datatools import imread, imsave
from data.conversion_tools import annotations2y


def combine_annot(b_save=True):
    ext = 100

    red = [255, 0, 0]
    blue = [0, 0, 255]
    
    folder_annots = '/home/lameeus/data/ghent_altar/input/hierachy/10_lamb/annotations/lameeus'
    folder_save = '/home/lameeus/data/ghent_altar/input/hierachy/10_lamb/annotations/kfold'


    
    # Clean image
    im_clean = imread('/home/lameeus/data/ghent_altar/input/hierachy/10_lamb/clean.png')
    im_white = 255*np.ones(shape=im_clean.shape, dtype=np.uint8)
    
    im_annot = np.copy(im_white)
    im_annot_clean = np.copy(im_clean)

    def apply_redblue(im, h0, h1, w0, w1, b0, b1):
        # red to red
        # rest to blue
        im[h0:h1, w0:w1, :][~b1, :] = blue
        im[h0:h1, w0:w1, :][b1, :] = red

    for i in range(6):
        im = imread(os.path.join(folder_annots, f'crop{i+1}_annotation_lm.png'))

        w0, w1, h0, h1 = lst_get[i]()
        
        im_crop = im[ext:-ext, ext:-ext, :]
        y_crop = annotations2y(im_crop, thresh=.9)
        b1 = y_crop[..., 1].astype(bool)
        
        if 0:
            plt.figure()
            plt.imshow(im_crop)
            plt.show()
        
        if b_save:
            im_annot_i = np.copy(im_white)
            im_annot_clean_i = np.copy(im_clean)

            apply_redblue(im_annot_i, h0, h1, w0, w1, ~b1, b1)
            apply_redblue(im_annot_clean_i, h0, h1, w0, w1, ~b1, b1)
            
            imsave(os.path.join(folder_save, f'annot_{i+1}.png'), im_annot_i)
            imsave(os.path.join(folder_save, f'annot_clean_{i+1}.png'), im_annot_clean_i)

        apply_redblue(im_annot, h0, h1, w0, w1, ~b1, b1)
        apply_redblue(im_annot_clean, h0, h1, w0, w1, ~b1, b1)
    
    plt.figure()
    plt.imshow(im_annot)
    plt.show()

    if b_save:
        imsave(os.path.join(folder_save, 'annot_comb.png'), im_annot)
        imsave(os.path.join(folder_save, 'annot_clean_comb.png'), im_annot_clean)
    
    return


def get_borders1():
    # ear
    w0 = 200
    w1 = 500
    h0 = 600
    h1 = 900
    
    return w0, w1, h0, h1


def get_borders2():
    # scalp
    w0 = 700
    w1 = 1000
    h0 = 250
    h1 = 500
    
    return w0, w1, h0, h1


def get_borders3():
    # scalp2
    w0 = 1050
    w1 = 1350
    h0 = 150
    h1 = 400
    
    return w0, w1, h0, h1


def get_borders4():
    # mouth
    w0 = 900
    w1 = 1250
    h0 = 1050
    h1 = 1400
    
    return w0, w1, h0, h1


def get_borders5():
    # shoulder (see xray)
    w0 = 1590
    w1 = 1940
    h0 = 1020
    h1 = 1420
    
    return w0, w1, h0, h1


def get_borders6():
    # right of scalp
    w0 = 1250
    w1 = 1600
    h0 = 450
    h1 = 700
    
    return w0, w1, h0, h1


if __name__ == '__main__':
    combine_annot()
