from data.example_splits import panel19withoutRightBot
from data.main import annotations2y, y2bool_annot
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



    y_tr, y_te = panel19withoutRightBot(y_annot)
    
    b = False
    if b:
        concurrent([a.get('clean'), y_annot, y_tr, y_te], ['clean', 'annotation', 'train', 'test'])
        
    n = neuralNet0(mod=mod)
    
    from datasets.training_examples import get_train19_topleft
    
    train_data = get_train19_topleft(mod=mod)
    x = train_data.get_x_train()
    y_tr = train_data.get_y_train()
    
    x_te = train_data.get_x_test()
    y_te = train_data.get_y_test()
    
    n.train(x, y_tr, (x_te, y_te))
    
    mod_set = get_mod_set(mod)
    x_img = img2array(a.get(mod_set))
    
    o = n.inference(x_img)[..., 0]
    
    b = False
    if b:
        # Model
        t = Threshholding()
        
        t.method = local_thresholding
        
        o = t.inference(a.get('clean'))
    
    # plotting results
    concurrent([a.get('clean'), o], ['clean', 'prediction'])
    
    print('Done')
    