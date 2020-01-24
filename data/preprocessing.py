import numpy as np


def zero2one(x):
    x_array = np.asarray(x)
    
    x_rescaled = np.empty(shape=x_array.shape, dtype=np.float16)
    
    if x_array.max() > 255:
        x_rescaled[...] = x_array/65535.
    elif x_array.max() > 1:
        x_rescaled[...] = x_array/255.
    else:
        x_rescaled[...] = x_array.copy()
        
    return x_rescaled


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