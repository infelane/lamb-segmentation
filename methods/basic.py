import numpy as np

import skimage.filters as filters

from data.preprocessing import img2batch, batch2img
from preprocessing.image import get_flow


class Base(object):
    def inference(self, x_img):
        return self.method(x_img)

class Threshholding(Base):
    
    def method(self, x_img):
        
        thresh = np.mean(x_img)
        gray = np.mean(x_img, axis=2)
        
        return np.greater_equal(gray, thresh)
    

class NeuralNet(Base):
    def __init__(self, model):
        self.model = model
    
    def method(self, x_img):
        
        x = img2batch(x_img)
        
        y = self.model.predict(x)
        return y[0]
    
    def train(self, x, y, validation=None, epochs=20, class_weight=None):
        steps_per_epoch = 100
        
        flow_tr = get_flow(batch2img(x), batch2img(y))
        
        flow_va = get_flow(*map(batch2img, validation)) if (validation is not None) else None
        
        self.get_model().fit_generator(flow_tr, epochs=epochs, steps_per_epoch=100, validation_data=flow_va, class_weight=class_weight, validation_steps=steps_per_epoch//10)
        
    def get_model(self):
        return self.model


def local_thresholding(x_img, ext:int=200):
    gray = np.mean(x_img, axis=2)
    
    x_threshold = filters.threshold_local(gray, block_size=ext*2+1)
    
    # Debugging:
    if 1:
        from plotting import concurrent
        concurrent([gray, x_threshold])
    
    return np.greater_equal(gray, x_threshold)
