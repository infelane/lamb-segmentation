import os

import numpy as np

import skimage.filters as filters

try:
    from tensorflow.keras.preprocessing.image import NumpyArrayIterator
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
except (ImportError, ModuleNotFoundError):
    from keras.preprocessing.image import NumpyArrayIterator
    from keras.callbacks import ModelCheckpoint, TensorBoard
# from tensorflow.keras.preprocessing.image import NumpyArrayIterator
# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from data.conversion_tools import batch2img
from neuralNetwork.results import inference
from preprocessing.image import get_flow
from data.preprocessing import rescale0to1


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
    epoch=0
    def __init__(self, model, w_ext=0,
                 norm_x=False):

        self.model = model
        
        self.w_ext = w_ext

        self.norm_x = norm_x
        
    def load(self, folder, epoch):
        self.model.load_weights(os.path.join(folder, f'w_{epoch}.h5'))
        self.epoch = epoch
        
    def predict(self, x_img, w=None):
        if self.norm_x:
            print('normalising input')
            x = rescale0to1(x_img)
        else:
            x = x_img

        return self.method(x, w=w)
    
    def method(self, x_img, w=None):
        if self.norm_x:
            print('normalising input')
            x = rescale0to1(x_img)
        else:
            x = x_img

        return inference(self.model, x, w=w, w_ext=self.w_ext)
        
    def train(self, xy, validation=None, epochs=1, verbose=1, info='scratch',
              steps_per_epoch=100
              ):
        """
        
        :param xy: Can be either tuple of (x, y) or Keras Generator
        :param validation:
        :param epochs:
        :param verbose:
        :param info:
        :param steps_per_epoch:
        :return:
        """

        if self.norm_x:
            print('normalising input')

            assert isinstance(xy, tuple)

            x, y = xy

            x = rescale0to1(x)

            xy = (x, y)
        
        def get_flow_xy(xy):
            if isinstance(xy, tuple):
                x, y = map(batch2img, xy)
                
                flow = get_flow(batch2img(x), batch2img(y))
                return flow
            
            elif isinstance(xy, (NumpyArrayIterator, )):
                return xy
                
            else:
                raise TypeError(f'Unkown type for xy: {type(xy)}')

        flow_tr = get_flow_xy(xy)
        
        flow_va = get_flow_xy(validation) if (validation is not None) else None

        folder_base = 'C:/Users/admin/Data/ghent_altar/' if os.name == 'nt' else '/scratch/lameeus/data/ghent_altar/'

        folder_checkpoint = os.path.join(folder_base, 'net_weight', info)
        filepath_checkpoint = os.path.join(folder_checkpoint, 'w_{epoch}.h5')
        folder_tensorboard = os.path.join(folder_base, 'logs', info)
        
        if not os.path.exists(folder_checkpoint):
            os.makedirs(folder_checkpoint)
            
        checkpoint = ModelCheckpoint(filepath_checkpoint, save_weights_only=False)
        # Only save the graph if it's first epoch training the model
        tensorboard = TensorBoard(folder_tensorboard, write_graph=self.epoch==0)
        callbacks = [checkpoint, tensorboard]
        
        self.get_model().fit_generator(flow_tr,
                                       initial_epoch=self.epoch, epochs=self.epoch+epochs,
                                       steps_per_epoch=steps_per_epoch,
                                       validation_data=flow_va, validation_steps=steps_per_epoch//10,
                                       verbose=verbose, callbacks=callbacks)
        self.epoch += epochs
        
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
