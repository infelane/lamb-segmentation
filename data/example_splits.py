"""

"""

import numpy as np


def panel19withoutRightBot(y):
    """
    There is an interest for this with Shaoguang's paper comparison
    """
    
    assert y.shape[:2] == (1401, 2101)
    
    y_tr = np.zeros(shape = y.shape, dtype=y.dtype)
    y_te = np.zeros(shape = y.shape, dtype=y.dtype)
    
    reg_rightbot = np.zeros(shape=y.shape[:2], dtype=bool)
    reg_rightbot[1000:, 1600:]=True

    y_tr[~reg_rightbot, ...] = y[~reg_rightbot, ...]
    y_te[reg_rightbot, ...] = y[reg_rightbot, ...]
    
    return y_tr, y_te
