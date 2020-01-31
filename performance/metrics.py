import numpy as np

from neuralNetwork.import_keras import K

# smooth = 1e-6


def accuracy_with0(y_true, y_pred, verbose=False):
    
    tp, tn, fp, fn = _tp_tn_fp_fn(y_true, y_pred)
    acc = (tn + tp) / (tn + tp + fn + fp)
    
    if verbose:
        print('accuracy_with0 = {}'.format(K.eval(acc)))
    
    return acc


def jaccard_with0(y_true, y_pred, verbose=False):
    """
    Jaccard is tp / (tp + all falses)
    """
    
    tp, tn, fp, fn = _tp_tn_fp_fn(y_true, y_pred)
    
    if tp + fp + fn == 0:
        jaccard = 1
    else:
        jaccard = tp/(tp + fp + fn)
    
    if verbose:
        print('jaccard_with0 = {}'.format(K.eval(jaccard)))
    
    return jaccard


def _tp_tn_fp_fn(y_true, y_pred):
    
    true0 = y_true[..., 0]
    true1 = y_true[..., 1]

    arg_pred = K.argmax(y_pred, axis=-1)

    pred0 = K.cast(K.equal(arg_pred, 0), y_pred.dtype)
    pred1 = K.cast(K.equal(arg_pred, 1), y_pred.dtype)

    tp = _sum_and(pred1, true1)  # positive is class 1 and it is correct
    tn = _sum_and(pred0, true0)  # negative is class 0 and it is correct
    fp = _sum_and(pred1, true0)  # class 1 is predicted but it is incorrect
    fn = _sum_and(pred0, true1)  # class 0 is predicted but it is incorrect
    
    return tp, tn, fp, fn


def _sum_and(a, b):
    """ with a and b binary, it does count the amount of AND gives true"""

    # return K.sum(a * b)
    return K.sum(K.cast(K.greater(a, 0), K.floatx()) *  K.cast(K.greater(b, 0), K.floatx()))
