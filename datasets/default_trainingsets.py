import numpy as np

from .training import TrainData

from datasets.example_splits import panel19withoutRightBot, panel13withoutRightBot
from data.conversion_tools import annotations2y, img2array
from data.modalities import get_mod_set
from datasets.examples import get_19hand, get_13zach, get_10lamb, get_10lamb_kfold


def get_10lamb_old(mod):
    """
    Face of the lamb on panel 10
    :param mod:
    :return:
    """
    
    img_x, img_y = xy_from_df(get_10lamb(), mod)
    
    train_data = TrainData(img_x, img_y, np.zeros(img_y.shape))

    return train_data


def get_10lamb_6patches(mod):
    img_x = x_from_df(get_10lamb(), mod)

    df_kfold_annot = get_10lamb_kfold()

    lst_names = [f'annot_{i+1}' for i in range(6)]
    y_img_lst = list(map(annotations2y, df_kfold_annot.get(lst_names)))
    
    return KFoldTrainData(img_x, y_img_lst)


def get_13(mod, debug=False):
    """
    Prophet Zachary
    :param mod:
    :return:
    """

    img_x, img_y = xy_from_df(get_13zach(), mod)

    train_data = TrainData(img_x, img_y, np.zeros(shape=img_y.shape))

    return train_data


def get_1319(mod):
    img_x13, img_y13 = xy_from_df(get_13zach(), 5)
    img_x19, img_y19 = xy_from_df(get_19hand(), 5)

    img_x = [img_x13, img_x19]
    img_y = [img_y13, img_y19]

    # No test data
    train_data = TrainData(img_x, img_y, [np.zeros(shape=img_y_i.shape) for img_y_i in img_y])

    return train_data


def get_13botleftshuang(mod, n_per_class=80, debug=False):
    """
    John the evangelist see Journal 2020 S. Huang
    :param mod:
    :param n_per_class:
    :return:
    """
    # TODO n_per_class according to the paper is 40...
    
    img_x, img_y = xy_from_df(get_13zach(), mod)
    
    _, img_y_2 = panel13withoutRightBot(img_y)

    shape = img_x.shape[:2]
    idx = np.mgrid[:shape[0], :shape[1]]
    
    # Should be transposed
    idx2 = np.transpose(idx, (1, 2, 0))
    idx_flat = idx2.reshape((-1, 2))
    
    b_flat0 = img_y_2[..., 0].flatten().astype(bool)
    b_flat1 = img_y_2[..., 1].flatten().astype(bool)
    
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

    y_tr = np.zeros(img_y_2.shape)
    y_te = np.zeros(img_y_2.shape)

    y_tr[b_tr, :] = img_y_2[b_tr, :]
    y_te[~b_tr, :] = img_y_2[~b_tr, :]
   
    if debug:
        import matplotlib.pyplot as plt
        plt.imshow(np.sum(y_tr, axis=-1))
        plt.show()
    
    # More like double checking:
    assert (np.count_nonzero(y_tr[..., 0]), np.count_nonzero(y_tr[..., 1])) == (n_per_class, n_per_class)

    assert np.count_nonzero(img_y_2) == np.count_nonzero(y_tr) + np.count_nonzero(y_te)

    train_data = TrainData(img_x, y_tr, y_te)

    return train_data


def get_train19_topleft(mod):
    """
    Excluded
    """
    
    img_x, img_y = xy_from_df(get_19hand(), mod)
    
    y_tr, y_te = panel19withoutRightBot(img_y)
    
    train_data = TrainData(img_x, y_tr, y_te)
    
    return train_data


def get_19SE_shuang(mod, n_per_class = 80, debug=False):
    """
    John the evangelist see Journal 2020 S. Huang
    :param mod:
    :param n_per_class:
    :return:
    """
    
    img_x, img_y_full = xy_from_df(get_19hand(), mod)
    _, img_y = panel19withoutRightBot(img_y_full)
    
    img_y_tr, img_y_te = _split_train_n_per_class(img_y, n_per_class)

    if debug:
        import matplotlib.pyplot as plt
        plt.imshow(np.sum(img_y_tr, axis=-1))
        plt.show()

    train_data = TrainData(img_x, img_y_tr, img_y_te)

    return train_data


def xy_from_df(df, mod):
    img_x = x_from_df(df, mod)
    img_y = annotations2y(df.get('annot'))
    
    return img_x, img_y


def x_from_df(df, mod):
    mod_set = get_mod_set(mod)
    img_x = np.concatenate(list(map(img2array, df.get(mod_set))), axis=-1)
    return img_x
    
    
def _split_train_n_per_class(img_y, n_per_class):
    
    assert len(img_y.shape) == 3
    
    shape = img_y.shape[:2]
    
    idx1 = np.mgrid[:shape[0], :shape[1]]
    # Should be transposed
    idx2 = np.transpose(idx1, (1, 2, 0))
    idx_flat = idx2.reshape((-1, 2))

    b_flat0 = img_y[..., 0].flatten().astype(bool)
    b_flat1 = img_y[..., 1].flatten().astype(bool)
    
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
    
    img_y_tr = np.zeros(img_y.shape)
    img_y_te = np.zeros(img_y.shape)

    img_y_tr[b_tr, :] = img_y[b_tr, :]
    img_y_te[~b_tr, :] = img_y[~b_tr, :]
    
    # More like double checking:
    assert (np.count_nonzero(img_y_tr[..., 0]), np.count_nonzero(img_y_tr[..., 1])) == (n_per_class, n_per_class)

    assert np.count_nonzero(img_y) == np.count_nonzero(img_y_tr) + np.count_nonzero(img_y_te)

    return img_y_tr, img_y_te


class KFoldTrainData(object):
    def __init__(self, x_img:np.ndarray, y_img_lst:list):

        assert isinstance(x_img, np.ndarray)
        assert all(isinstance(y_img_i, np.ndarray) for y_img_i in y_img_lst)
        
        self.x_img = x_img
        self.y_img_lst = y_img_lst

        self.n = len(y_img_lst)
        
    def k_split_i(self, i:int):
        assert isinstance(i, (int, np.int_))
        assert 0 <= i < self.n, f'i should be in range [0, {self.n}-1]. i = {i}'

        y_te = self.y_img_lst[i]
        y_tr_lst = self.y_img_lst[:i] + self.y_img_lst[i+1:]
        
        y_tr = np.any(y_tr_lst, axis=0)

        train_data = TrainData(self.x_img, y_tr, y_te)
        return train_data
    
    def get_train_data_all(self):
        
        y_all = np.any(self.y_img_lst, axis=0)
        
        train_data = TrainData(self.x_img, y_all, np.zeros(y_all.shape, dtype=y_all.dtype))
        return train_data
