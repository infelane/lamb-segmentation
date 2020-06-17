import numpy as np

from datasets.default_trainingsets import get_13botleftshuang, get_19SE_shuang, \
    get_1319, get_10lamb_old, get_19SE_shuang_crack
from main_general import get_training_data

from preprocessing.image import get_flow


def load_data(data_name, n_per_class, seed=None):
    def _data10nat():
        # Sparse annotations
        train_data = get_10lamb_old(5)
        
        from datasets.examples import get_10lamb
        from data.conversion_tools import annotations2y
        train_data.y_tr = annotations2y(get_10lamb().get("annot_tflearning"))
        
        return train_data
    
    if data_name == '13botright':
        train_data = get_13botleftshuang(5, n_per_class=n_per_class)
    
    elif data_name == '19botright':
        train_data = get_19SE_shuang(5, n_per_class=n_per_class, seed=seed)
    
    elif data_name == '19botrightcrack':
        train_data = get_19SE_shuang_crack(5, n_per_class=n_per_class)
    
    elif data_name == '19botrightcrack3':
        train_data = get_19SE_shuang_crack(5, n_per_class=n_per_class, n_outputs=3)
    
    elif data_name == '1319':
        train_data = get_1319(5)
        
    elif data_name == '1319botright':
        
        from datasets.training import TrainData
    
        a13 = load_data("13botright", n_per_class)
        a19 = load_data("19botright", n_per_class)
    
        img_x = [a13.get_x_train(), a19.get_x_train()]
        img_y_train = [a13.get_y_train(), a19.get_y_train()]
        img_y_val = [a13.get_y_test(), a19.get_y_test()]
    
        train_data = TrainData(img_x, img_y_train, img_y_val)
    
    elif data_name.split('_')[-1] == '10':
        
        # Sparse annotations
        train_data = get_10lamb_old(5)
    
    elif data_name.split('_')[-1] == '101319':
        from datasets.default_trainingsets import xy_from_df, get_13zach, get_19hand, get_10lamb, TrainData
        
        img_x10, img_y10 = xy_from_df(get_10lamb(), 5)
        img_x13, img_y13 = xy_from_df(get_13zach(), 5)
        img_x19, img_y19 = xy_from_df(get_19hand(), 5)
        
        img_x = [img_x10, img_x13, img_x19]
        img_y = [img_y10, img_y13, img_y19]
        
        # No test data
        train_data = TrainData(img_x, img_y, [np.zeros(shape=img_y_i.shape) for img_y_i in img_y])
    
    elif data_name.split('_')[-1] == '10nat':
        train_data = _data10nat()
    
    elif data_name.split('_')[-1] == '10nat1319':
        from datasets.default_trainingsets import TrainData
        
        train_data10 = _data10nat()
        train_data1319 = get_1319(5)
        
        img_x = [train_data10.get_x_train()] + train_data1319.get_x_train()
        img_y = [train_data10.get_y_train()] + train_data1319.get_y_train()
        
        train_data = TrainData(img_x,
                               img_y, [np.zeros(shape=img_y_i.shape) for img_y_i in img_y])
    
    else:
        raise ValueError(data_name)
    
    from data.preprocessing import rescale0to1
    train_data.x = rescale0to1(train_data.x)
    
    return train_data
