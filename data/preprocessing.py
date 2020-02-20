import numpy as np


def rescale0to1(x):
    x_rescaled = np.empty(shape=x.shape, dtype=np.float32)  # float16 is annoying to plot...
    
    if x.max() > 255:
        x_rescaled[...] = x/65535.
    elif x.max() > 1:
        x_rescaled[...] = x/255.
    else:
        x_rescaled[...] = x.copy()
        
    return x_rescaled
