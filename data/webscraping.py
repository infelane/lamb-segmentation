"""
In some cases http://closertovaneyck.kikirpa.be/ghentaltarpiece/
seems to have better quality images

Why not scrape them, this should be allowed:
http://closertovaneyck.kikirpa.be/ghentaltarpiece/#home/sub=copyright
"""

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import requests
from io import BytesIO


def preset_10lamb_rgb():
    i_w_0 = 137
    i_w_1 = 143
    
    i_h_0 = 47
    i_h_1 = 51
    
    return i_w_0, i_w_1, i_h_0, i_h_1


def preset_10lamb_ir():
    i_w_0 = 137
    i_w_1 = 143
    
    i_h_0 = 47
    i_h_1 = 51
    
    return i_w_0, i_w_1, i_h_0, i_h_1


def preset_13zachary(mod = None):
    i_w_0 = 58
    i_w_1 = 76
    
    i_h_0 = 7
    i_h_1 = 34

    if mod == 'irr':
        i_w_0 = 14
        i_w_1 = 18

        i_h_0 = 1
        i_h_1 = 8
        
    elif mod == 'xray':
        i_w_0 = 28
        i_w_1 = 38

        i_h_0 = 3
        i_h_1 = 18

    return i_w_0, i_w_1, i_h_0, i_h_1


def preset_19hand_ir():
    i_w_0 = 33
    i_w_1 = 46
    
    i_h_0 = 89
    i_h_1 = 98

    return i_w_0, i_w_1, i_h_0, i_h_1


def preset_19hand_rgb():
    i_w_0 = 33//4
    i_w_1 = 46//4
    
    i_h_0 = 89//4
    i_h_1 = 98//4
    
    return i_w_0, i_w_1, i_h_0, i_h_1


def preset_19hand_xray():
    i_w_0 = 33 // 2
    i_w_1 = 46 // 2
    
    i_h_0 = 89 // 2
    i_h_1 = 98 // 2
    
    return i_w_0, i_w_1, i_h_0, i_h_1


def get_im(panel_nr=10, modality='rgb', resolution_scale=0, b_plot=True, b_grid=True):
    """
    
    Parameters
    ----------
    panel_nr
    modality
    resolution_scale:
        0 if highest possible (17)
    b_plot

    Returns
    -------

    """

    modality = modality.lower()

    if panel_nr in [10, 13, 19]:
        pass
    else:
        raise NotImplementedError()

    if resolution_scale == 0:
        f_res = 17
    elif isinstance(resolution_scale, int):
        f_res = resolution_scale
    else:
        raise NotImplementedError()
        
    if panel_nr == 10 and modality == 'rgb':
        i_w_0, i_w_1, i_h_0, i_h_1 = preset_10lamb_rgb()

    elif panel_nr == 10 and modality == 'ir':
        i_w_0, i_w_1, i_h_0, i_h_1 = preset_10lamb_ir()
    
    if panel_nr == 13:
        i_w_0, i_w_1, i_h_0, i_h_1 = preset_13zachary(modality)
        
    elif panel_nr == 19 and modality == 'ir':
        i_w_0, i_w_1, i_h_0, i_h_1 = preset_19hand_ir()
    elif panel_nr == 19 and modality == 'rgb':
        i_w_0, i_w_1, i_h_0, i_h_1 = preset_19hand_rgb()
    elif panel_nr == 19 and modality == 'xray':
        i_w_0, i_w_1, i_h_0, i_h_1 = preset_19hand_xray()
        
    else:
        raise NotImplementedError()
    
    i_w_range = np.arange(i_w_0, i_w_1+1)
    i_h_range = np.arange(i_h_0, i_h_1+1)
    
    if panel_nr in [13, 19]:
        if modality == 'ir':
            f_mod = 'IR-HI-BTL'
        elif modality == 'rgb':
            f_mod = 'VIS-MR-BT'
        elif modality == 'xray':
            f_mod = 'XR-MR-BTL'
        elif modality == 'irr':
            f_mod = 'IRR-MR-BTL'
        else: raise NotImplementedError()
        f_url = lambda: f'http://data.closertovaneyck.be/ec2/tiles/00-{panel_nr}-{f_mod}//{f_res}/{i_w_i}_{i_h_i}.jpg'
    else:

        if modality.lower() == 'rgb':
            f_mod = f'{panel_nr}MPVISB0001'
            pass
        elif modality.lower() == 'ir':
            f_mod = f'{panel_nr}MCIRPB0001'
            pass
        else:
            raise NotImplementedError()
        
        f_url = lambda: f'http://data.closertovaneyck.be/legacy/tiles/{f_mod}//{f_res}/{i_w_i}_{i_h_i}.jpg'
    
    im_lst = {}
    for i_w_i in i_w_range:
        
        print(f'{i_w_i - i_w_0} / {i_w_1-i_w_0}')
        
        for i_h_i in i_h_range:
            url = f_url()
            im = open_url_image(url)

            h_i, w_i = im.shape[:2]
        
            im_lst[(i_w_i, i_h_i)] = [(h_i, w_i), im]
        
    w_lst = []
    for i_w_i in i_w_range:
        # info patch; shape patch; width
        w_lst.append(im_lst[(i_w_i, i_h_0)][0][1])
    w = sum(w_lst)
    
    h_lst = []
    for i_h_i in i_h_range:
        # info patch; shape patch; height
        h_lst.append(im_lst[(i_w_0, i_h_i)][0][0])
    h = sum(h_lst)
    
    # add one zero to the front
    w_cumsum = np.pad(np.cumsum(w_lst), (1, 0), mode='constant')
    h_cumsum = np.pad(np.cumsum(h_lst), (1, 0), mode='constant')
    
    im_example = im_lst[list(im_lst.keys())[0]][1]
    dtype_im = im_example.dtype
    
    shape_stitched = (h, w) + im_example.shape[2:]
    im_stitched = np.zeros(shape_stitched, dtype=dtype_im)
    for i_w, i_w_i in enumerate(i_w_range):
        for i_h, i_h_i in enumerate(i_h_range):
            
            h0 = h_cumsum[i_h]
            h1 = h_cumsum[i_h+1]
            w0 = w_cumsum[i_w]
            w1 = w_cumsum[i_w+1]
            im_stitched[h0:h1, w0:w1, ...] = im_lst[(i_w_i, i_h_i)][1]
            
    if b_plot:
        plt.figure()
        plt.imshow(im_stitched)
        if b_grid:
    
            grid = np.zeros((h, w, 4), dtype=dtype_im)  # alpha channel
            # change the colour
            cyan = [0, 255, 255]
            grid[..., :3] = cyan
    
            for w_i in w_cumsum[1:-1]:
                grid[:, w_i, 3] = 255
            for h_i in h_cumsum[1:-1]:
                grid[h_i, :, 3] = 255
            
            plt.imshow(grid) # To show different patches on top of stitched
            
    plt.show()
    
    return im_stitched
    
    
def open_url_image(url):
    assert requests.get(url).status_code < 400, f'{url}'
    
    response = requests.get(url)

    img = Image.open(BytesIO(response.content))
    
    return np.array(img)
    

if __name__ == '__main__':
    """
    Test environment
    """
    im = get_im(panel_nr=10, modality='ir', b_grid=False)
