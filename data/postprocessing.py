import numpy as np


def img_to_uint8(x):
    
    assert x.min() >= 0
    assert isinstance(x, np.ndarray)
    
    if x.max() <= 1:
        x_rescaled = (x*255)
    elif x.max() <= 255:
        x_rescaled = x
    else:
        x_rescaled = np.floor(x/256.)
    
    return x_rescaled.astype(np.uint8)
