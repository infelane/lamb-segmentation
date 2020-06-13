from math import ceil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, jaccard_similarity_score, cohen_kappa_score

from data.conversion_tools import batch2img
from performance.metrics import accuracy_with0, jaccard_with0
from neuralNetwork.import_keras import K


def optimal_test_thresh(y_pred, y_tr, y_te, verbose=1, d_thresh = .1, thresh0=0, thresh1=1):
    """
    
    :param y_pred:
    :param y_tr:
    :param y_te:
    :param verbose:
        0 show nothing
        1 show everything
        2 show minimal
    :param d_thresh:
    :param thresh0:
    :param thresh1:
    :return:
    """

    ### Preprocessing
    if y_tr is None:
        y_pred, y_te = map(batch2img, (y_pred, y_te))
    else:
        (y_pred, _, y_te) = map(batch2img, (y_pred, y_tr, y_te))

    # Filter nonzero:
    y_te, y_pred = filter_non_zero(y_te, y_pred)

    ### Precalculations
    y_te_argmax = np.argmax(y_te, axis=-1)

    ### Performance with threshold = .5
    y_pred_thresh_argmax = get_y_pred_thresh_argmax(y_pred, .5)
    acc_te, jacc_te, kappa_te = _get_scores(y_te_argmax, y_pred_thresh_argmax)
    
    if verbose == 1:
        print('Theshold = .5:')
        print(f'acc = {acc_te:.4f}\t jaccard = {jacc_te:.4f}\t kappa = {kappa_te:.4f}')

    thresh_range = np.arange(thresh0 + d_thresh, thresh1, d_thresh) # exlcuding thresh0 and thresh1

    lst_data = []
    
    for thresh in thresh_range:
        y_pred_thresh_argmax = get_y_pred_thresh_argmax(y_pred, thresh)

        acc_te, jacc_te, kappa_te = _get_scores(y_te_argmax, y_pred_thresh_argmax)

        if verbose==1:
            print(f'Thresh: {thresh}')
            print(f'acc = {acc_te:.4f}\t jaccard = {jacc_te:.4f}\t kappa = {kappa_te:.4f}')

        lst_data.append({'thresh':thresh, 'accuracy':acc_te, 'jaccard':jacc_te, 'kappa':kappa_te})

    df = pd.DataFrame(lst_data)
    
    if verbose == 1:
        df.plot('thresh', ['accuracy', 'jaccard', 'kappa'])
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    
    kappa_max = df['kappa'].max()
    te_thresh = df['thresh'][df['kappa'].idxmax()]
    
    if verbose:
        print(f'maximum of kappa: {kappa_max}')
        print(f'optimal (test) threshold: {te_thresh}')

    return te_thresh


def optimal_test_thresh_equal_distribution(y_true, y_pred, mask_true=True):
    """
    
    :param y_true: Ground truth (GT) to get distribution from
    :param y_pred: prediction to match distribution to
    :param mask_true: take subset of prediction that is included in GT
    :return:
    """

    assert y_true.shape[-1] == y_pred.shape[-1] == 2
    
    from data.conversion_tools import y2bool_annot
    
    def _get_distribution(y):
    
        list_n = np.sum(y, axis=tuple(np.arange(len(y.shape) - 1)))
        
        list_p = list_n/np.sum(list_n)

        return list_p

    p_true = _get_distribution(y_true)
    print('p_true:', p_true)

    ### Find threshold to match distribution
    y_pred1 = y_pred[y2bool_annot(y_true), 1] if mask_true else  y_pred[..., 1]
    n_pred = np.size(y_pred1)
    y_pred1_sorted = np.sort(y_pred1.flatten())
    i_thresh = ceil((1-p_true[1])*n_pred)

    thresh = y_pred1_sorted[i_thresh]
    
    y_pred1_thresholded = np.greater_equal(y_pred1, thresh)
    y_pred_thresholded = np.stack([1-y_pred1_thresholded, y_pred1_thresholded], axis=-1)
    
    # Assertion?
    p_pred_thresholded = _get_distribution(y_pred_thresholded)
    
    print('p_pred_thresholded:', p_pred_thresholded)
    
    # Inclusive
    return thresh


def test_thresh_incremental(y_pred, y_tr=None, y_te=None, n=1, verbose=0, thresh0=0, thresh1=1, n_incr=10):
    
    assert n_incr >= 3
    
    for _ in range(n):
        d_thresh = (thresh1 - thresh0) / float(n_incr)
        test_thresh = optimal_test_thresh(y_pred, y_tr, y_te, verbose=verbose, d_thresh=d_thresh, thresh0=thresh0, thresh1=thresh1)
        
        thresh0 = test_thresh - d_thresh
        thresh1 = test_thresh + d_thresh
        
    return test_thresh


def test_thresh_optimal(y_pred, y_test, metric='kappa'):

    # Precalculations
    ## Remove non-important values
    y_test_non0, y_pred_non0 = filter_non_zero(y_test, y_pred)

    y_te_argmax = np.argmax(y_test_non0, axis=-1)

    def m(t):

        y_pred_thresh_argmax = get_y_pred_thresh_argmax(y_pred_non0, t)

        acc_te, jacc_te, kappa_te = _get_scores(y_te_argmax, y_pred_thresh_argmax)

        if metric == 'kappa':
            mm = kappa_te
        elif metric == 'jaccard':
            mm = jacc_te
        elif metric == 'accuracy':
            mm = acc_te
        else:
            raise ValueError(metric)

        return mm

    t0_old = 0.
    t1_old = 1.

    v0_old = m(t0_old)
    v1_old = m(t1_old)

    t_lst_max = .5
    v_max = m(t_lst_max)

    delta = .5

    while delta > .01:

        t0_new = (t0_old + t_lst_max)/2.
        t1_new = (t_lst_max + t1_old)/2.

        v0_new = m(t0_new)
        v1_new = m(t1_new)

        t_lst = [t0_old, t0_new, t_lst_max, t1_new, t1_old]
        v_lst = [v0_old, v0_new, v_max, v1_new, v1_old]

        i_max, v_max = max(enumerate(v_lst), key=lambda x: x[1])

        if i_max == len(t_lst) - 1:  # Last element is biggest
            i_max += -1  # Just act as if index before is best.

        elif i_max == 0:  # First element is biggest
            i_max += 1  # Just act as if index after is best.

        assert 1 <= i_max <= len(t_lst) - 1 - 1

        t_lst_max = t_lst[i_max]

        t0_old = t_lst[i_max - 1]
        t1_old = t_lst[i_max + 1]

        v0_old = v_lst[i_max - 1]
        v1_old = v_lst[i_max + 1]

        delta = (t1_old - t0_old) / 2

    return t_lst_max


def filter_non_zero(y_true, y_pred):
    b_nonzero = np.any(y_true, axis=-1)

    y_true_filter = y_true[b_nonzero, ...]
    y_pred_filter = y_pred[b_nonzero, ...]

    return y_true_filter, y_pred_filter


def get_y_pred_thresh(y_pred, thresh):
    y_pred1 = np.greater_equal(y_pred[..., 1], thresh).astype(np.float32)
    y_pred_thresh = np.stack([1 - y_pred1, y_pred1], axis=-1)
    return y_pred_thresh
    

def get_y_pred_thresh_argmax(y_pred, thresh):
    y_pred_thresh = get_y_pred_thresh(y_pred, thresh)

    y_pred_thresh_argmax = np.argmax(y_pred_thresh, axis=-1)

    return y_pred_thresh_argmax
    
    
def _get_scores(y_te_argmax, y_pred_thresh_argmax):
    acc_te = accuracy_score(y_te_argmax, y_pred_thresh_argmax)
    jacc_te = jaccard(y_te_argmax, y_pred_thresh_argmax)
    kappa_te = cohen_kappa_score(y_te_argmax, y_pred_thresh_argmax)

    return acc_te, jacc_te, kappa_te


def jaccard(y_true, y_pred):
    AoO = np.count_nonzero(np.logical_and(y_true, y_pred))
    AoU = np.count_nonzero(np.logical_or(y_true, y_pred))

    if AoU == 0:
        Warning(f'AoU is 0')
        return 1
    else:
        return AoO / AoU
