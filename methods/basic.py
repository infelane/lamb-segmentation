import numpy as np

import skimage.filters as filters

from data.preprocessing import zero2one


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
        
        x_pre = zero2one(x_img)
        
        x_img_input = np.reshape(x_pre, newshape=(1, ) + x_pre.shape)
        
        y = self.model.predict(x_img_input)
        return y[0]
    
    def train(self, x, y_tr, validation=None, epochs=20):
        
        from preprocessing.image import get_flow
        flow_tr = get_flow(x[0], y_tr[0])
        
        self.get_model().fit_generator(flow_tr, epochs=epochs, steps_per_epoch=100, validation_data=validation)
        # self.get_model().fit(x, y_tr, epochs=epochs,
        #                validation_data=validation)
        
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
