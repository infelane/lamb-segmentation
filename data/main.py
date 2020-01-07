import numpy as np

from .preprocessing import zero2one


def annotations2y(img_annot_rgb, thresh: float = 1.):
    shape = np.shape(img_annot_rgb)
    shape_class = [shape[0], shape[1], 2]
    
    img_class = np.zeros(shape=shape_class, dtype=np.uint8)
    
    assert thresh >= 0 and thresh <= 1, f'thresh has to be in [0, 1]: {thresh}'
    
    img_rescale = zero2one(img_annot_rgb)
    
    r0 = np.greater_equal(1 - thresh, img_rescale[:, :, 0])
    r1 = np.greater_equal(img_rescale[:, :, 0], thresh)
    g0 = np.greater_equal(1 - thresh, img_rescale[:, :, 1])
    g1 = np.greater_equal(img_rescale[:, :, 1], thresh)
    b0 = np.greater_equal(1 - thresh, img_rescale[:, :, 2])
    b1 = np.greater_equal(img_rescale[:, :, 2], thresh)
    
    red = r1 * g0 * b0  # loss
    blue = r0 * g0 * b1  # background
    
    img_class[blue, 0] = 1
    img_class[red, 1] = 1
    
    return img_class


def y2bool_annot(y):
    return np.any(y, axis=-1)