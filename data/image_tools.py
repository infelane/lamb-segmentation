import numpy as np


class ImgExt(object):
    """ the image_ext will now be slightly bigger"""
    
    def __init__(self, img, ext, edge='mirror', w=0):
        """
        :param img:
        :param ext:
        :param edge: mirror: mirror the edges, grey: edges set to 0.5
        """
        
        shape = np.shape(img)
        
        assert isinstance(ext, (tuple, int))
        if type(ext) is int:  # convert to tuple
            ext = (ext, ext)

        if type(ext) is tuple:
            assert len(ext) == 2
            assert ext[0] >= 0
            assert ext[1] >= 0
            
            shape_ext = [shape[0] + ext[0] + ext[1] + w, shape[1] + ext[0] + ext[1] + w, shape[2]]
        
        elif type(ext) is int:
            assert ext >= 0
            shape_ext = [shape[0] + 2 * ext + w, shape[1] + 2 * ext + w, shape[2]]
        
        else:
            raise TypeError
        
        if edge == 'grey':
            self._img_ext = np.ones(shape_ext, dtype=img.dtype) * 0.5
        else:
            self._img_ext = np.zeros(shape_ext, dtype=img.dtype)
        self.ext = ext
        
        if type(ext) is tuple:
            self._img_ext[ext[0]:ext[0] + shape[0], ext[0]:ext[0] + shape[1], :] = img
            if edge == 'mirror':

                self._img_ext[:ext[0], ext[0]:shape[1] + ext[0], :] = np.flip(img[1:ext[0] + 1, :, :], axis=0)
                self._img_ext[shape[0] + ext[0]:, ext[0]:shape[1] + ext[0], :] = np.flip(
                    img[shape[0] - ext[1] - w - 1:shape[0] - 1, :, :], axis=0)
                
                self._img_ext[:, :ext[0], :] = np.flip(self._img_ext[:, ext[0] + 1: 2 * ext[0] + 1, :], axis=1)
                self._img_ext[:, shape[1] + ext[0]:, :] = np.flip(
                    self._img_ext[:, shape[1] + ext[0] - ext[1] - w - 1:shape[1] + ext[0] - 1, :],
                    axis=1)
        
        elif type(ext) is int:
            if ext > 0:
                self._img_ext[ext: shape[0] + ext, ext: + shape[1] + ext, :] = img
                if edge == 'mirror':
                    
                    self._img_ext[:ext, ext:shape[1] + ext, :] = np.flip(img[1:ext + 1, :, :], axis=0)
                    self._img_ext[shape[0] + ext:, ext:shape[1] + ext, :] = np.flip(
                        img[shape[0] - ext - w - 1:shape[0] - 1, :, :], axis=0)
                    
                    self._img_ext[:, :ext, :] = np.flip(self._img_ext[:, ext + 1: 2 * ext + 1, :], axis=1)
                    self._img_ext[:, shape[1] + ext:, :] = np.flip(
                        self._img_ext[:, shape[1] - w - 1:shape[1] + ext - 1, :],
                        axis=1)
            
            else:
                self._img_ext[...] = img
    
    def get_crop(self, i_h, i_w, w):
        ext = self.ext
        if type(ext) is tuple:
            return self()[i_h: i_h + w + ext[0] + ext[1], i_w: i_w + w + ext[0] + ext[1], :]
        elif type(ext) is int:
            return self()[i_h: i_h + w + 2 * ext, i_w: i_w + w + 2 * ext, :]
    
    def get_extended(self):
        return self._img_ext
    
    def __call__(self, *args, **kwargs):
        return self._img_ext


class DataInference(object):
    def __init__(self, img, w=10):
        self.shape = np.shape(img)
        
        assert len(self.shape) <= 3
        
        self.w = w
    
    def img_to_x(self, img: np.array, ext: int or tuple = 0) -> object:
        w = self.w
        
        shape_in = np.shape(img)
        
        n_h = _ceildiv(self.shape[0], w)
        n_w = _ceildiv(self.shape[1], w)
        
        self.n_h = n_h
        self.n_w = n_w
        
        if type(ext) is tuple:
            assert len(ext) == 2
            shape = (n_h * n_w, w + ext[0] + ext[1], w + ext[0] + ext[1], shape_in[2])
        
        elif type(ext) is int:
            shape = (n_h * n_w, w + 2 * ext, w + 2 * ext, shape_in[2])
        
        else:
            raise TypeError('ext is expected to be tuple or integer')
        
        x = np.empty(shape=shape, dtype=np.float16)
        
        img_ext = ImgExt(img, ext=ext, w=w)
        
        for i_h in range(n_h):
            for i_w in range(n_w):
                x[i_h * n_w + i_w, :, :, :] = img_ext.get_crop(i_h * self.w, i_w * self.w, self.w)
        
        return x
    
    def y_to_img(self, y, ext=0):
        w = self.w
        n_h = self.n_h
        n_w = self.n_w
        
        shape_big = (self.shape[0] + w, self.shape[1] + w, np.shape(y)[3])
        
        img_y = np.ones(shape=shape_big) * 0.5
        
        if ext == 0:
            for i_h in range(n_h):
                for i_w in range(n_w):
                    img_y[i_h * w:(i_h + 1) * w, i_w * w:(i_w + 1) * w, :] = y[i_h * n_w + i_w, :, :, :]
        
        else:
            for i_h in range(n_h):
                for i_w in range(n_w):
                    img_y[i_h * w:(i_h + 1) * w, i_w * w:(i_w + 1) * w, :] = y[i_h * n_w + i_w, ext:-ext, ext:-ext, :]
        
        # return img_y
        return img_y[:self.shape[0], :self.shape[1], :]


def _ceildiv(a, b):
    return -(-a // b)
