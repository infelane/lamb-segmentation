import numpy as np

from .training import TrainData

from data.example_splits import panel19withoutRightBot, panel13withoutRightBot
from data.conversion_tools import annotations2y
from data.modalities import get_mod_set
from data.preprocessing import img2array
from datasets.examples import get_19hand, get_13zach, get_10lamb


def get_10lamb_all(mod):
    """
    Face of the lamb on panel 10
    :param mod:
    :return:
    """
    
    df = get_10lamb()

    mod_set = get_mod_set(mod)
    x_img = img2array(df.get(mod_set))

    y_img = annotations2y(df.get('annot'))

    train_data = TrainData(x_img, y_img, y_img)

    return train_data


def get_13botleftshuang(mod, n_per_class = 80, debug=False):
    """
    John the evangelist see Journal 2020 S. Huang
    :param mod:
    :param n_per_class:
    :return:
    """
    # TODO n_per_class according to the paper is 40...

    df = get_13zach()

    mod_set = get_mod_set(mod)
    
    x_img = img2array(df.get(mod_set))

    y_img = annotations2y(df.get('annot'))
    
    _, y_img2 = panel13withoutRightBot(y_img)

    shape = x_img.shape[:2]
    idx = np.mgrid[:shape[0], :shape[1]]
    
    # Should be transposed
    idx2 = np.transpose(idx, (1, 2, 0))
    idx_flat = idx2.reshape((-1, 2))
    
    b_flat0 = y_img2[..., 0].flatten().astype(bool)
    b_flat1 = y_img2[..., 1].flatten().astype(bool)
    
    def shuffle(arr):
        arr_shuffle = np.copy(arr)
        np.random.seed(314)
        np.random.shuffle(arr_shuffle)
        return arr_shuffle
    
    idx_shuffled = shuffle(idx_flat)
    b_shuffle0 = shuffle(b_flat0)
    b_shuffle1 = shuffle(b_flat1)
    
    idx0 = idx_shuffled[b_shuffle0][:n_per_class]
    idx1 = idx_shuffled[b_shuffle1][:n_per_class]

    b_tr = np.zeros(shape, dtype=bool)

    for i, j in idx0:
        b_tr[i, j] = True
        
    for i, j in idx1:
        b_tr[i, j] = True

    y_tr = np.zeros(y_img2.shape)
    y_te = np.zeros(y_img2.shape)

    y_tr[b_tr, :] = y_img2[b_tr, :]
    y_te[~b_tr, :] = y_img2[~b_tr, :]
   
    if debug:
        import matplotlib.pyplot as plt
        plt.imshow(np.sum(y_tr, axis=-1))
        plt.show()
    
    # More like double checking:
    assert (np.count_nonzero(y_tr[..., 0]), np.count_nonzero(y_tr[..., 1])) == (n_per_class, n_per_class)

    assert np.count_nonzero(y_img2) == np.count_nonzero(y_tr) + np.count_nonzero(y_te)

    train_data = TrainData(x_img, y_tr, y_te)

    return train_data


def get_train19_topleft(mod):
    """
    Excluded
    """
    
    a = get_19hand()
    
    mod_set = get_mod_set(mod)
    x_img = img2array(a.get(mod_set))
    
    y_img = annotations2y(a.get('annot'))
    
    y_tr, y_te = panel19withoutRightBot(y_img)
    
    train_data = TrainData(x_img, y_tr, y_te)
    
    return train_data


def get_19SE_shuang(mod, n_per_class = 80, debug=True):
    """
    John the evangelist see Journal 2020 S. Huang
    :param mod:
    :param n_per_class:
    :return:
    """

    df = get_19hand()

    mod_set = get_mod_set(mod)

    # TODO rename to img_x
    img_x = img2array(df.get(mod_set))
    _, img_y = panel19withoutRightBot(annotations2y(df.get('annot')))
    
    img_y_tr, img_y_te = _split_train_n_per_class(img_y, n_per_class)

    if debug:
        import matplotlib.pyplot as plt
        plt.imshow(np.sum(img_y_tr, axis=-1))
        plt.show()

    train_data = TrainData(img_x, img_y_tr, img_y_te)

    return train_data


def _split_train_n_per_class(y_img, n_per_class):
    
    assert len(y_img.shape) == 3
    
    shape = y_img.shape[:2]
    
    idx1 = np.mgrid[:shape[0], :shape[1]]
    # Should be transposed
    idx2 = np.transpose(idx1, (1, 2, 0))
    idx_flat = idx2.reshape((-1, 2))

    b_flat0 = y_img[..., 0].flatten().astype(bool)
    b_flat1 = y_img[..., 1].flatten().astype(bool)
    
    def _shuffle(arr):
        # always the same shuffle!
        arr_shuffle = np.copy(arr)
        np.random.seed(314)
        np.random.shuffle(arr_shuffle)
        return arr_shuffle

    idx_shuffled = _shuffle(idx_flat)
    b_shuffle0 = _shuffle(b_flat0)
    b_shuffle1 = _shuffle(b_flat1)
    
    idx0 = idx_shuffled[b_shuffle0][:n_per_class]
    idx1 = idx_shuffled[b_shuffle1][:n_per_class]
    
    b_tr = np.zeros(shape, dtype=bool)
    for i, j in idx0:
        b_tr[i, j] = True
    for i, j in idx1:
        b_tr[i, j] = True
    
    y_img_tr = np.zeros(y_img.shape)
    y_img_te = np.zeros(y_img.shape)

    y_img_tr[b_tr, :] = y_img[b_tr, :]
    y_img_te[~b_tr, :] = y_img[~b_tr, :]
    
    # More like double checking:
    assert (np.count_nonzero(y_img_tr[..., 0]), np.count_nonzero(y_img_tr[..., 1])) == (n_per_class, n_per_class)

    assert np.count_nonzero(y_img) == np.count_nonzero(y_img_tr) + np.count_nonzero(y_img_te)

    return y_img_tr, y_img_te
