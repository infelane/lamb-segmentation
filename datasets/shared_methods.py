import os

from data.datatools import imread
from plotting import concurrent


class Data(dict):
    _imgs = {}
    def __init__(self):
        
        assert self._imgs
        
        super().__init__(self._imgs)
    
    def get(self, name=None):
        """
        If name == None: returns all images
        """
        
        if name is not None:
            if isinstance(name, str):
                assert name in self._imgs
                return self._imgs[name]
            elif isinstance(name, (list, tuple)):
                
                l = []
                for name_i in name:
                    assert name_i in self._imgs
                    l.append(self._imgs[name_i])
                return l

            else:
                raise KeyError(f"Don't know how to handle: {name}")

        else:
            # Return all images
            return self._imgs

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
                
                img = imread(path)

                self._imgs[root] = img

        super().__init__()
