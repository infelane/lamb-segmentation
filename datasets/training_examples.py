import numpy as np

from .training import TrainData

from data.example_splits import panel19withoutRightBot, panel13withoutRightBot
from data.conversion_tools import annotations2y
from data.modalities import get_mod_set
from data.preprocessing import img2array
from datasets.examples import get_19hand, get_13zach


def get_train19_topleft(mod):
    """
    Excluded
    """
    
    a = get_19hand()

    mod_set = get_mod_set(mod)
    x_img = img2array(a.get(mod_set))
    
    y_img = annotations2y(a.get('annot'))
    
    y_tr, y_te = panel19withoutRightBot(y_img)
    
    trainData = TrainData(x_img, y_tr, y_te)
    
    return trainData


def get_13botleftshuang(mod, n_per_class = 80, debug=False):
    """
    John the evangelist see Journal 2020 S. Huang
    :param mod:
    :param n_per_class:
    :return:
    """

    idx_shuffled = get_13zach()

    mod_set = get_mod_set(mod)
    
    x_img = img2array(idx_shuffled.get(mod_set))

    y_img = annotations2y(idx_shuffled.get('annot'))
    
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

    trainData = TrainData(x_img, y_tr, y_te)

    return trainData
    