import os

from .shared_methods import DataFolder

def get_10lamb():

    folder = 'C:/Users/Laurens_laptop_w/data/ghent_altar/input/10_lamb' if os.name == 'nt' else '/home/lameeus/data/ghent_altar/input/hierachy/10_lamb'
    d = DataFolder(folder)
    return d


def get_10lamb_kfold():
    folder = 'C:/Users/Laurens_laptop_w/data/ghent_altar/input/10_lamb/annotations/kfold' if os.name == 'nt' else'/home/lameeus/data/ghent_altar/input/hierachy/10_lamb/annotations/kfold'
    return DataFolder(folder)

def get_13zach():
    d = DataFolder('/home/lameeus/data/ghent_altar/input/hierachy/13_small')
    return d


def get_19hand():
    d = DataFolder('/home/lameeus/data/ghent_altar/input/hierachy/19_small')
    return d
