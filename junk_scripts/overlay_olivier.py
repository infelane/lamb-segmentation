import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def colour2single(im):
    # Float is needed for when subtracting 128 (else overflow as it is uint8)
    c = np.mean(np.abs(im.astype(float) - 128), axis=-1, keepdims=True)  # negative gray, then abs, then average
    c_norm = c / c.max()

    return c_norm


def cmap_func(b):

        if 0:   # Boring single colour, but can still be used ofcourse
            # cyan
            overlay_colour = np.reshape([0, 255, 255], (1, 1, 3))
            return overlay_colour

        elif 1:     # Use a more fancy colourmap
            assert b.shape[-1] == 1
            # cm.jet: from blue to yellow to red
            # Ignore alpha
            # Working in UINT8
            return cm.jet(b[:, :, 0])[..., :3]*255

        else:
            raise NotImplementedError()


def overlay(im_orig, b, alpha_max=.5):
    """
    alpha_max: 1 means b==1 is full colour, .5 means, b==1 is only half colour, half original pixel value
    """

    assert 0 <= alpha_max <= 1

    im_overlay = np.copy(im_orig).astype(np.float)  # better safe than sorry (although normally auto conversion here)
    # Scale overlay colour with b as "transparancy".
    im_overlay = alpha_max*b * cmap_func(b) + (1 - alpha_max*b) * im_overlay
    im_overlay = im_overlay.astype(np.uint8)

    return im_overlay


def script_overlay(im_orig, im_colour):

    c = colour2single(im_orig)

    im_overlay = overlay(im_orig, c)
    return im_overlay


if __name__ == '__main__':

    # TODO Adjust to where images are located
    f = 'C:/Users/admin/downloads'

    im0 = np.array(Image.open(os.path.join(f, 'o0.jpg')))   # The rgb image
    # im1 = np.array(Image.open(os.path.join(f, 'o1.jpg')))   # "square"
    im2 = np.array(Image.open(os.path.join(f, 'o2.jpg')))   # gradient map (with the grey background)
    im3 = np.array(Image.open(os.path.join(f, 'o3.jpg')))   # reduced gradient map

    im_overlay2 = script_overlay(im0, im2)
    im_overlay3 = script_overlay(im0, im3)

    if 1:
        plt.imshow(im_overlay2)
        plt.show()

    plt.imshow(im_overlay3)
    plt.show()

    # Saving results
    im = Image.fromarray(im_overlay2)
    im.save(os.path.join(f, 'overlay2.png'))

    im = Image.fromarray(im_overlay3)
    im.save(os.path.join(f, 'overlay3.png'))
