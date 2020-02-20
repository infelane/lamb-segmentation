import os

import numpy as np
from PIL import Image


def imread(path):
    im = np.array(Image.open(path))
    return im
    
    
def imsave(path, array, b_check_duplicate=True):
    
    if b_check_duplicate:
        if os.path.exists(path):
            feedback = input(f"file exists ({path}).\nType 'Y' if you want to overwrite it").lower()
            if feedback != 'y':
                print(f'Not saved: {path}')
                return

    Image.fromarray(array).save(path)
    return
