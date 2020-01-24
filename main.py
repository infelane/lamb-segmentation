import numpy as np
import matplotlib.pyplot as plt

from data.example_splits import panel19withoutRightBot
from data.conversion_tools import annotations2y, y2bool_annot
from data.modalities import get_mod_set
from data.preprocessing import img2array
from datasets.examples import get_19hand
from methods.basic import Threshholding, local_thresholding
from plotting import concurrent
from methods.examples import neuralNet0



if __name__ == '__main__':
    mod = 'all'
    
    ### Data
    a = get_19hand()
    b = False
    if b:
        a.plot()
    
    ### Training/Validation data
    img_y = a.get('annot')
    y = annotations2y(img_y)
    y_annot = y2bool_annot(y)
    
    b = False
    if b:
        y_annot_tr, y_annot_te = panel19withoutRightBot(y_annot)
    
        concurrent([a.get('clean'), y_annot, y_annot_tr, y_annot_te], ['clean', 'annotation', 'train annot', 'test annot'])
        
    n = neuralNet0(mod=mod)
    
    from datasets.training_examples import get_train19_topleft, get_13botleftshuang
    
    if 0:
        train_data = get_train19_topleft(mod=mod)
    else:
        train_data = get_13botleftshuang(mod=mod)

    # TODO normalise inputs This seems to be super important...
    # train_data.x = (1/255. * train_data.x).astype(np.float16)
    # train_data.x = (255. * train_data.x).astype(np.float16)
    
    x = train_data.get_x_train()
    y_tr = train_data.get_y_train()
    
    x_te = train_data.get_x_test()
    y_te = train_data.get_y_test()
    
    if 1:
        from preprocessing.image import get_flow
        flow_tr = get_flow(x[0], y_tr[0])

        from preprocessing.image import get_class_imbalance

        class_weight = get_class_imbalance(flow_tr)
    
    if 0:
        from neuralNetwork.optimization import find_learning_rate
        if 0:
            find_learning_rate(n.get_model(), (x, y_tr), class_weight)
            
        # Optimal Lr ~= 1e-4
        else: lr_opt = find_learning_rate(n.get_model(), flow_tr, class_weight)
    else:

        lr_opt = 1e-1
    
    print(f'Optimal expected learning rate: {lr_opt}')
    
    n = neuralNet0(mod=mod, lr=lr_opt)
    n.train(x, y_tr, (x_te, y_te), class_weight=class_weight, epochs=1000)
    
    mod_set = get_mod_set(mod)
    x_img = img2array(x)
    
    o = n.inference(x_img)[..., 0]
    
    b = False
    if b:
        # Model
        t = Threshholding()
        
        t.method = local_thresholding
        
        o = t.inference(a.get('clean'))
    
    # plotting results
    concurrent([a.get('clean'), o], ['clean', 'prediction'])
    
    ### Evaluation
    
    
    
    print('Done')
