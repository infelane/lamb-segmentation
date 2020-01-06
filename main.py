from data.main import annotations2y, y2bin_annot
from datasets.examples import get_19hand
from methods.basic import Threshholding, local_thresholding
from plotting import concurrent

if __name__ == '__main__':
    ### Data
    a = get_19hand()
    if 0:
        a.plot()
    
    ### Training/Validation data
    img_y = a.get('annot')
    y = annotations2y(img_y)
    y_annot = y2bin_annot(y)

    concurrent([a.get('clean'), y_annot], ['clean', 'annotation'])
    
    # Model
    t = Threshholding()
    
    t.method = local_thresholding
    
    o = t.inference(a.get('clean'))
    
    # plotting results
    concurrent([a.get('clean'), o], ['clean', 'prediction'])
    
    