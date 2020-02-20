import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def open_image(path):
    return np.asarray(Image.open(path))


def save_image(im, path):
    if 0:
        Image.fromarray(im).save(path)


def main1():
    
    im = open_image('/scratch/lameeus/data/ghent_altar/input/hierarchy/mask_pred_comb.png')
    im_clean = open_image('/home/lameeus/data/ghent_altar/input/hierachy/10_lamb/clean.png')
    
    im_annot = np.ones(shape=im_clean.shape, dtype=im_clean.dtype)
    clean_annot = np.copy(im_clean)
    
    b0 = np.equal(im, 0)
    b1 = np.equal(im, 255)

    blue = [0, 0, 255]
    red =[255, 0, 0]
    
    im_annot[b0, :] = blue
    im_annot[b1, :] = red
    clean_annot[b0, :] = blue
    clean_annot[b1, :] = red
    
    plt.imshow(im_annot)
    
    plt.imshow(clean_annot)

    save_image(im_annot, '/scratch/lameeus/data/ghent_altar/input/hierarchy/annot.png')
    save_image(clean_annot, '/scratch/lameeus/data/ghent_altar/input/hierarchy/clean_annot.png')

    return


if __name__ == '__main__':
    main1()
