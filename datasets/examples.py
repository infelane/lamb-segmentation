import os

from .shared_methods import DataFolder

base_folder = 'C:/Users/admin/OneDrive - ugentbe/data' if os.name == 'nt' else '/home/lameeus/data/ghent_altar/input/hierarchy'


def get_10lamb():

    folder = os.path.join(base_folder, '10_lamb')
    # folder = 'C:/Users/Laurens_laptop_w/data/ghent_altar/input/10_lamb' if os.name == 'nt' else '/home/lameeus/data/ghent_altar/input/hierarchy/10_lamb'
    d = DataFolder(folder)
    return d


def get_10lamb_kfold():
    if os.name == 'nt':
        folder = os.path.join(base_folder, '10_lamb/laptop_data/annotations/kfold')
    else:
        folder = os.path.join(base_folder, '10_lamb/annotations/kfold')

    return DataFolder(folder)


def get_13zach():

    d = DataFolder(os.path.join(base_folder, '13_small'))
    return d


def get_19hand():
    d = DataFolder(os.path.join(base_folder, '19_small'))
    return d
