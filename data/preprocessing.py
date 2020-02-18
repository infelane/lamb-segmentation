import numpy as np


def rescale0to1(x):
    x_array = np.asarray(x)
    
    x_rescaled = np.empty(shape=x_array.shape, dtype=np.float16)
    
    if x_array.max() > 255:
        x_rescaled[...] = x_array/65535.
    elif x_array.max() > 1:
        x_rescaled[...] = x_array/255.
    else:
        x_rescaled[...] = x_array.copy()
        
    return x_rescaled
