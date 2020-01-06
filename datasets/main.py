import os


import numpy as np
from PIL import Image

from plotting import concurrent

class Data(object):
    imgs = {}
    def __init__(self):
        
        assert self.imgs
        raise ValueError('')
    
    def get(self, name=None):
        """
        Returns all images
        """
        
        if name is not None:
            assert name in self.imgs
            return self.imgs[name]
        
        else:
            # Return all images
            return self.imgs
        
    def plot(self):
        imgs = self.get()

        keys = imgs.keys()
        img_list = [imgs[key] for key in keys]
    
        concurrent(img_list, titles=keys)
        
        
        
class DataFolder(Data):
    def __init__(self, folder):
    
        for file in os.listdir(folder):
            root, ext = os.path.splitext(file)
            if ext.lower() in ['.png', '.tif', '.jpg']:
                path = os.path.join(folder, file)
        
                img = Image.open(path)
                
                self.imgs[root] = img
