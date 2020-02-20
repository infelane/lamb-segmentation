import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import resize

from data.conversion_tools import annotations2y


def get_image(path):
    img = Image.open(path)
    return np.array(img)


def save_img(path, img):
    Image.fromarray(img).save(path)


def get_crop(img):
    if img.shape[:2] == (1945, 1218):
        h0, h1 = 1623, 1923
        w0, w1 = 912, 1212
    elif img.shape[:2] == (1401, 2101):
        h0, h1 = 1083, 1383
        w0, w1 = 1740, 2040

    return img[h0:h1, w0:w1, ...]
    
    
def crop_left_bot_script():
    
    panel_nr = 19 # 13, 19
    modality = 'annot'
    modality = 'src'
    
    folder = f'/home/lameeus/data/ghent_altar/input/hierachy/{panel_nr}_small'
    if modality == 'annot':
        path_mod = os.path.join(folder, 'annot' + '.tif')
    elif modality == 'src':
        folder_src = f'/scratch/lameeus/data/ghent_altar/output/hierarchy/{panel_nr}_small/fancy'
        path_mod = os.path.join(folder_src, f'{panel_nr}_3_src-kf.jpg')
        
    path_clean = os.path.join(folder, 'clean.tif')

    img_clean = get_image(path_clean)[..., :3]
    img_mod = get_image(path_mod)
    
    img_clean = get_crop(img_clean)
    if modality == 'annot':
        img_mod = get_crop(img_mod)
    elif modality == 'src':
        img_mod = resize(img_mod, (300, 300))

    y_mod = annotations2y(img_mod, thresh=.7)
    
    if modality == 'src':
        plt.imshow(y_mod[..., 0])
    
    cyan = [0, 255, 255]
    green = [0, 127, 0]
    lime = [50, 205, 50]
    gray = [127, 127, 127]
    white = [255, 255, 255]
    black = [0, 0, 0]
    
    b1 = y_mod[..., 1].astype(bool)
    
    img_clean_y = np.copy(img_clean)
    img_clean_y[b1, :] = cyan
    # background
    def grayer(img, img_clean, b1):
        frac = .5
        img[~b1, :] = (frac*img_clean[~b1, :] + (1-frac)*np.asarray(gray)).astype(img.dtype)
    def darker(img, img_clean, b1):
        frac = .5
        img[~b1, :] = (frac*img_clean[~b1, :] + (1-frac)*np.asarray(black)).astype(img.dtype)

    grayer(img_clean_y, img_clean, b1)
    # darker(img, img_clean, b1)
    
    plt.imshow(img_clean_y)
    plt.show()
    
    if 0:
        path_save = f'/scratch/lameeus/data/ghent_altar/output/hierarchy/{panel_nr}_small/fancy/shuang_{modality}.png'
        if os.path.exists(path_save) == False:
            save_img(path_save, img_clean_y)
        else:
            print(f"didn't save: {path_save}")

    return img_clean_y

    
if __name__ == '__main__':
    crop_left_bot_script()