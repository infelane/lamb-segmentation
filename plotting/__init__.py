import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def concurrent(imgs, titles=None):
    
    if titles is not None:
        assert len(imgs) == len(titles)
        titles_list = list(titles)
    
    n = len(imgs)
    n_w = int(np.ceil(n ** .5))
    n_h = int(np.ceil(n / n_w))
    
    plt.figure()
    for i, img in enumerate(imgs):
        if isinstance(img, Image.Image):
            img_copy = np.asarray(img)
        else:
            img_copy = img
        
        if i == 0:
            ax1 = plt.subplot(n_h, n_w, i + 1)
        else:
            plt.subplot(n_h, n_w, i + 1, sharex=ax1, sharey=ax1)
        
        vmax = 1 if img_copy.dtype == bool else 255
        
        if len(img_copy.shape) == 2:
            plt.imshow(img_copy, cmap='gray', vmin=0, vmax=vmax)
        else:
            plt.imshow(img_copy, vmin=0, vmax=vmax)
            
        if titles is not None:
            plt.title(titles_list[i])
    
    plt.show()