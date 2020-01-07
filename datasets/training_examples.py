import numpy as np

from .training import TrainData

from data.example_splits import panel19withoutRightBot
from data.main import annotations2y
from data.modalities import get_mod_set
from data.preprocessing import img2array
from datasets.examples import get_19hand


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
