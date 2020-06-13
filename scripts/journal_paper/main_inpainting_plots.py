import os

import matplotlib.pyplot as plt
import numpy as np

from data.datatools import imread, imsave
from data.conversion_tools import detect_colour

from plotting import concurrent


# Replace paint loss in original with inpaint
def inpaint_replacer(im_orig, b_mask, im_inpaint):

    assert im_orig.shape[:2] == b_mask.shape == im_inpaint.shape[:2]

    im_new = im_orig.copy()
    im_new[b_mask, :] = im_inpaint[b_mask, :]
    return im_new


if __name__ == '__main__':

    f = "C:/Users/admin/Downloads"

    # Read original
    im_orig = imread("C:/Users/admin/OneDrive - ugentbe/data/10_lamb/clean.png")

    # Read paint loss detection
    b_mask = detect_colour(imread("C:/Users/admin/OneDrive - ugentbe/data/images_paper/1319_10nat_V3.png"), "cyan")
    if 0:
        plt.imshow(b_mask)

    # Read inpaint
    im_inpaint = imread(os.path.join(f, "inpainting_comb.jpg"))

    im_inpainted_new = inpaint_replacer(im_orig, b_mask, im_inpaint)
    plt.imshow(im_inpainted_new)

    concurrent([im_orig, im_inpainted_new, im_inpaint, b_mask], ['orig', 'new', 'old', 'mask'])

    # Save
    imsave(os.path.join(f, "inpaint_v2.png"), im_inpainted_new)

    # Crop?
    im_treated = imread(os.path.join(f, "restored_SR.jpg"))

    # Trying to rescale to similar colour pallet
    im_treated_recolour = im_treated.astype(float)
    im_treated_recolour = (im_treated_recolour - im_treated.mean((0, 1)))*im_inpainted_new.std((0, 1))/im_treated.std((0, 1)) + im_inpainted_new.mean((0, 1))
    im_treated_recolour = np.clip(im_treated_recolour, 0, 255)
    im_treated_recolour = im_treated_recolour.astype(np.uint8)

    imsave(os.path.join(f, "10treated2.png"), im_treated_recolour)

    l = []
    h0 = 150
    w0 = 550
    h1 = 1400
    w1 = 1300
    for im_i in [im_orig, im_inpainted_new, im_treated, im_treated_recolour]:
        l.append(im_i[h0:h1,w0:w1,:])

    concurrent(l)

    for name, l_i in zip(['10cropclean.png', '10cropvirt.png', '10croptreated.png', '10croptreated2.png'], l):
        imsave(os.path.join(f, name), l_i)
