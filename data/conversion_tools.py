import numpy as np

from .preprocessing import rescale0to1


def annotations2y(img_annot_rgb, thresh: float = 1.):
    shape = np.shape(img_annot_rgb)
    shape_class = [shape[0], shape[1], 2]
    
    img_class = np.zeros(shape=shape_class, dtype=np.uint8)
    
    assert thresh >= 0 and thresh <= 1, f'thresh has to be in [0, 1]: {thresh}'
    
    img_rescale = rescale0to1(img_annot_rgb)
    
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
    
    # List of annotated images
    if isinstance(y, list):
        return [y2bool_annot(y_i) for y_i in y]
    else:
        return np.any(y, axis=-1)


def img2array(img):
    
    if isinstance(img, list):
        return np.concatenate([img2array(img_i) for img_i in img], axis=-1)
    else:
        try:
            img_array = np.array(img)
        except:
            raise AssertionError(f'Should be image type: {type(img)}')

        if len(img_array.shape) == 2:
            return img_array.reshape(img_array.shape + (1, ))
        else:
            return img_array


def img2batch(x):
    
    assert (2 <= len(x.shape) <= 4), x.shape
    
    if len(x.shape) == 2:
        return x.reshape((1,) + x.shape + (1,))
    elif len(x.shape) == 3:
        return x.reshape((1,) + x.shape)
    else:
        return x


def batch2img(x):
    
    assert (len(x.shape) <= 4), x.shape
    
    if len(x.shape) == 4:
    
        assert x.shape[0] == 1, x.shape
        
        return x.reshape(x.shape[1:])
    else:
        return x
