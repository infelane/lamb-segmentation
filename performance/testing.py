import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, jaccard_similarity_score, cohen_kappa_score

from data.preprocessing import batch2img
from performance.metrics import accuracy_with0, jaccard_with0
from neuralNetwork.import_keras import K


def test(y_pred, y_tr, y_te):

    # preprocessing
    (y_pred, y_tr, y_te) = map(batch2img, (y_pred, y_tr, y_te))

    d_thresh = .1

    thresh_range = np.arange(d_thresh, 1, d_thresh)

    lst_thresh = []
    lst_acc = []
    lst_jacc = []

    def jaccard(y_true, y_pred):
        AoO = np.count_nonzero(np.logical_and(y_true, y_pred))
        AoU = np.count_nonzero(np.logical_or(y_true, y_pred))

        if AoU == 0:
            Warning(f'AoU is 0')
            return 1
        else:
            return AoO / AoU

    for thresh in thresh_range:
        print(f'Thresh: {thresh}')

        y_pred1 = np.greater_equal(y_pred[..., 1], thresh).astype(np.float32)
        y_pred_thresh = np.stack([1-y_pred1, y_pred1], axis=-1)

        # acc_tr = K.eval(accuracy_with0(y_tr, y_pred))
        # jacc_tr = K.eval(jaccard_with0(y_tr, y_pred))

        if 0:
            acc_te = K.eval(accuracy_with0(y_te, y_pred_thresh, False))
            jacc_te = K.eval(jaccard_with0(y_te, y_pred_thresh, False))
        else:
            acc_te = accuracy_score(y_te_argmax, y_pred_thresh_argmax)
            jacc_te = jaccard(y_te_argmax, y_pred_thresh_argmax)
            kappa_te = cohen_kappa_score(y_te_argmax, y_pred_thresh_argmax)

        print(f'acc = {acc_te:.4f}\t jaccard = {jacc_te:.4f}')

        lst_thresh.append(thresh)
        lst_acc.append(acc_te)
        lst_jacc.append(jacc_te)

        b_nonzero = y_te.sum(axis=-1).astype(bool)
        y_te_argmax = np.argmax(y_te[b_nonzero, :], axis=-1)
        y_pred_thresh_argmax = np.argmax(y_pred_thresh[b_nonzero, :], axis=-1)

    df = pd.DataFrame(zip(*[lst_thresh, lst_acc, lst_jacc]), columns=['thresh', 'accuracy', 'jaccard'])

    df.plot('thresh', ['accuracy', 'jaccard'])

    return -1
