import os

import numpy as np

import skimage.filters as filters

from keras.preprocessing.image import NumpyArrayIterator
from keras.callbacks import ModelCheckpoint, TensorBoard
# from tensorflow.keras.preprocessing.image import NumpyArrayIterator
# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from data.conversion_tools import batch2img
from neuralNetwork.results import inference
from preprocessing.image import get_flow


class Base(object):
    def predict(self, x_img):
        
        return self.method(x_img)

    # To be implemented
    def method(self, *args, **kwargs):
        raise NotImplementedError('Has to be implemented in child class')
    

class Threshholding(Base):
    
    def method(self, x_img):
        
        thresh = np.mean(x_img)
        gray = np.mean(x_img, axis=2)
        
        return np.greater_equal(gray, thresh)
    

class NeuralNet(Base):
    def __init__(self, model, w_ext=0):
        self.model = model
        
        self.w_ext = w_ext
        
    def predict(self, x_img, w=None):
        return self.method(x_img, w=w)
    
    def method(self, x_img, w=None):
        return inference(self.model, x_img, w=w, w_ext=self.w_ext)
        
    def train(self, xy, validation=None, epochs=1, verbose=1, info='scratch'):
        """
        
        :param xy: Can be either tuple of (x, y) or Keras Generator
        :param validation:
        :param epochs:
        :param class_weight:
        :param verbose:
        :return:
        """
        
        def get_flow_xy(xy):
            if isinstance(xy, tuple):
                x, y = map(batch2img, xy)
                
                flow = get_flow(batch2img(x), batch2img(y))
                return flow
            
            elif isinstance(xy, (NumpyArrayIterator, )):
                return xy
                
            else:
                raise TypeError(f'Unkown type for xy: {type(xy)}')
        
        steps_per_epoch = 100
        
        flow_tr = get_flow_xy(xy)
        
        flow_va = get_flow_xy(validation) if (validation is not None) else None

        folder_checkpoint =  os.path.join('/scratch/lameeus/data/ghent_altar/net_weight', info)
        filepath_checkpoint = os.path.join(folder_checkpoint, 'w_{epoch}.h5')
        folder_tensorboard = os.path.join('/scratch/lameeus/data/ghent_altar/logs/', info)
        
        if not os.path.exists(folder_checkpoint):
            os.makedirs(folder_checkpoint)
            
        checkpoint = ModelCheckpoint(filepath_checkpoint, save_weights_only=False)
        tensorboard = TensorBoard(folder_tensorboard)
        callbacks = [checkpoint, tensorboard]
        
        self.get_model().fit_generator(flow_tr, steps_per_epoch=steps_per_epoch,
                                       validation_data=flow_va, validation_steps=steps_per_epoch//10,
                                       epochs=epochs, verbose=verbose, callbacks=callbacks)
        
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
